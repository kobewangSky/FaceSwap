'''
Unified Face Swap API Server - Simplified Version
'''

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.videoswap import video_swap
from util.reverse2original import reverse2wholeimage
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
import os
import tempfile
import uuid
from typing import Optional
import sys
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
import queue
import json

app = FastAPI(title="Face Swap API", version="2.0.0")

# Global variables
model = None
face_app = None
transformer = None
transformer_Arcface = None
logoclass = None
spNorm = None
net = None

# Thread pool executor for concurrent processing
executor = ThreadPoolExecutor(max_workers=4)

# Real-time camera variables
camera_instances = {}

@dataclass
class VideoTask:
    task_id: str
    source_path: str
    video_path: str
    result_path: str
    temp_dir: str
    use_mask: bool
    status: str = "pending"
    progress: float = 0.0
    error: Optional[str] = None

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def preprocess_image_for_detection(image_path):
    """Preprocess image to improve face detection success rate"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"Original image shape: {img.shape}")
    
    # Resize image
    height, width = img.shape[:2]
    if width > 1920 or height > 1080:
        scale = min(1920/width, 1080/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height))
        print(f"Resized image shape: {img.shape}")
    
    # Enhance contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    img = cv2.merge([l, a, b])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    return img

def detect_faces_with_retry(face_app, img, crop_size, max_retries=3):
    """Face detection with retry mechanism"""
    original_thresh = face_app.det_thresh
    thresholds = [original_thresh, 0.4, 0.2, 0.1]
    
    for i, thresh in enumerate(thresholds[:max_retries]):
        print(f"Attempt {i+1}: Using detection threshold {thresh}")
        face_app.det_thresh = thresh
        
        try:
            result = face_app.get(img, crop_size)
            if result is not None:
                img_align_crop_list, mat_list = result
                if len(img_align_crop_list) > 0:
                    print(f"Success! Detected {len(img_align_crop_list)} faces with threshold {thresh}")
                    face_app.det_thresh = original_thresh  # Restore original threshold
                    return result
        except Exception as e:
            print(f"Detection attempt {i+1} failed: {e}")
            continue
    
    # Restore original threshold
    face_app.det_thresh = original_thresh
    print(f"All detection attempts failed. No faces detected.")
    return None

def process_source_image_for_identity(source_path, crop_size=224):
    """
    Process source image to extract identity features with automatic pre-crop detection
    Returns the identity latent vector
    """
    with torch.no_grad():
        # Load original image to check dimensions
        img_original = cv2.imread(source_path)
        if img_original is None:
            raise ValueError(f"Could not load source image: {source_path}")
        
        height, width = img_original.shape[:2]
        print(f"Source image dimensions: {width}x{height}")
        
        # Check if image is already cropped to face size (within ±50px of crop_size)
        crop_tolerance = 50
        is_pre_cropped = (abs(width - crop_size) <= crop_tolerance and 
                        abs(height - crop_size) <= crop_tolerance)
        
        if is_pre_cropped:
            print(f"Detected pre-cropped face image ({width}x{height}), skipping face detection...")
            # Resize to exact crop_size and use directly
            img_a_align_crop = cv2.resize(img_original, (crop_size, crop_size))
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop, cv2.COLOR_BGR2RGB))
        else:
            print("Using face detection pipeline...")
            # Preprocess image
            img_a_whole = preprocess_image_for_detection(source_path)
            
            # Use face detection with retry
            detection_result = detect_faces_with_retry(face_app, img_a_whole, crop_size)
            
            if detection_result is None:
                raise ValueError("No face detected in source image after multiple attempts")
            
            img_a_align_crop_list, _ = detection_result
            
            if len(img_a_align_crop_list) == 0:
                raise ValueError("No face detected in source image")
            
            # Use the first detected face
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop_list[0], cv2.COLOR_BGR2RGB))
        
        # Extract identity features
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
        
        img_id = img_id.cuda()
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)
        
        return latend_id

def reverse2wholeimage_realtime(b_align_crop_tenor_list, swaped_imgs, mats, crop_size, oriimg, logoclass, 
                               no_simswaplogo=False, pasring_model=None, norm=None, use_mask=False):
    """
    Modified version of reverse2wholeimage that returns the processed image instead of saving to disk
    """
    from util.reverse2original import SoftErosion, encode_segmentation_rgb, postprocess
    
    target_image_list = []
    img_mask_list = []
    if use_mask:
        smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).cuda()

    for swaped_img, mat, source_img in zip(swaped_imgs, mats, b_align_crop_tenor_list):
        swaped_img = swaped_img.cpu().detach().numpy().transpose((1, 2, 0))
        img_white = np.full((crop_size, crop_size), 255, dtype=float)

        # inverse the Affine transformation matrix
        mat_rev = np.zeros([2, 3])
        div1 = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        mat_rev[0][0] = mat[1][1] / div1
        mat_rev[0][1] = -mat[0][1] / div1
        mat_rev[0][2] = -(mat[0][2] * mat[1][1] - mat[0][1] * mat[1][2]) / div1
        div2 = mat[0][1] * mat[1][0] - mat[0][0] * mat[1][1]
        mat_rev[1][0] = mat[1][0] / div2
        mat_rev[1][1] = -mat[0][0] / div2
        mat_rev[1][2] = -(mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2]) / div2

        orisize = (oriimg.shape[1], oriimg.shape[0])
        if use_mask:
            source_img_norm = norm(source_img)
            source_img_512 = F.interpolate(source_img_norm, size=(512, 512))
            out = pasring_model(source_img_512)[0]
            parsing = out.squeeze(0).detach().cpu().numpy().argmax(0)
            vis_parsing_anno = parsing.copy().astype(np.uint8)
            tgt_mask = encode_segmentation_rgb(vis_parsing_anno)
            if tgt_mask.sum() >= 5000:
                target_mask = cv2.resize(tgt_mask, (crop_size, crop_size))
                target_image_parsing = postprocess(swaped_img, source_img[0].cpu().detach().numpy().transpose((1, 2, 0)), target_mask, smooth_mask)
                target_image = cv2.warpAffine(target_image_parsing, mat_rev, orisize)
            else:
                target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)[..., ::-1]
        else:
            target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)

        img_white = cv2.warpAffine(img_white, mat_rev, orisize)
        img_white[img_white > 20] = 255
        img_mask = img_white

        kernel = np.ones((40, 40), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        kernel_size = (20, 20)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)

        img_mask /= 255
        img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])

        if use_mask:
            target_image = np.array(target_image, dtype=np.float64) * 255
        else:
            target_image = np.array(target_image, dtype=np.float64)[..., ::-1] * 255

        img_mask_list.append(img_mask)
        target_image_list.append(target_image)

    # Composite the final image
    img = np.array(oriimg, dtype=np.float64)
    for img_mask, target_image in zip(img_mask_list, target_image_list):
        img = img_mask * target_image + (1 - img_mask) * img

    final_img = img.astype(np.uint8)
    if not no_simswaplogo:
        final_img = logoclass.apply_frames(final_img)
    
    return final_img

class RealtimeFaceSwapCamera:
    def __init__(self, source_image_path, crop_size=224, use_mask=True, camera_id=0):
        self.crop_size = crop_size
        self.use_mask = use_mask
        self.source_image_path = source_image_path
        self.camera_id = camera_id
        
        # Control variables
        self.running = False
        self.current_frame = None
        self.processing_time = 0
        self.fps = 0
        
        # Threading variables
        self.frame_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=2)
        self.latest_frame = None
        self.latest_processed = None
        self.camera_fps = 0
        self.process_fps = 0
        self.camera_frame_count = 0
        self.process_frame_count = 0
        
        # Thread references
        self.camera_thread_ref = None
        self.process_thread_ref = None
        
        # Initialize source identity
        self.setup_source_identity(source_image_path)
        
    def setup_source_identity(self, source_image_path):
        """Process source image to extract identity features"""
        print(f"Processing source image: {source_image_path}")
        
        try:
            self.latend_id = process_source_image_for_identity(source_image_path, self.crop_size)
            print("Source identity extracted successfully!")
                
        except Exception as e:
            print(f"Error in setup_source_identity: {e}")
            raise ValueError(f"Failed to process source image: {str(e)}")
    
    def process_frame(self, frame):
        """Process a single frame for face swap"""
        try:
            with torch.no_grad():
                # Detect faces
                detection_result = face_app.get(frame, self.crop_size)
                
                if detection_result is None:
                    return frame
                
                img_b_align_crop_list, b_mat_list = detection_result
                
                if len(img_b_align_crop_list) == 0:
                    return frame
                
                swap_result_list = []
                b_align_crop_tenor_list = []
                
                for b_align_crop in img_b_align_crop_list:
                    b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB))[None,...].cuda()
                    swap_result = model(None, b_align_crop_tenor, self.latend_id, None, True)[0]
                    swap_result_list.append(swap_result)
                    b_align_crop_tenor_list.append(b_align_crop_tenor)
                
                # Use the modified function that returns the image instead of saving
                output_frame = reverse2wholeimage_realtime(
                    b_align_crop_tenor_list, swap_result_list, b_mat_list, 
                    self.crop_size, frame, logoclass, 
                    False, pasring_model=net, 
                    use_mask=self.use_mask, norm=spNorm
                )
                
                return output_frame
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame
    
    def camera_capture_thread(self):
        """Camera capture thread"""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"Could not open camera {self.camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        camera_start_time = time.time()
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            self.latest_frame = frame.copy()
            
            self.camera_frame_count += 1
            camera_elapsed = time.time() - camera_start_time
            self.camera_fps = self.camera_frame_count / camera_elapsed if camera_elapsed > 0 else 0
            
            try:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame, block=False)
                else:
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass
            
            time.sleep(0.01)
        
        cap.release()
    
    def frame_processing_thread(self):
        """Frame processing thread"""
        process_start_time = time.time()
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                start_time = time.time()
                processed_frame = self.process_frame(frame)
                self.processing_time = time.time() - start_time
                
                self.process_frame_count += 1
                process_elapsed = time.time() - process_start_time
                self.process_fps = self.process_frame_count / process_elapsed if process_elapsed > 0 else 0
                
                self.latest_processed = processed_frame
                
                try:
                    if not self.result_queue.full():
                        self.result_queue.put(processed_frame, block=False)
                    else:
                        try:
                            self.result_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.result_queue.put(processed_frame, block=False)
                except queue.Full:
                    pass
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing thread: {e}")
                continue
    
    def get_latest_frame(self):
        """Get the latest processed frame"""
        try:
            if not self.result_queue.empty():
                result_frame = self.result_queue.get_nowait()
            elif self.latest_processed is not None:
                result_frame = self.latest_processed
            elif self.latest_frame is not None:
                result_frame = self.latest_frame
            else:
                return None
            
            # Add status information
            display_frame = result_frame.copy()
            
            cv2.putText(display_frame, f'Camera FPS: {self.camera_fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f'Process FPS: {self.process_fps:.1f}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f'Process Time: {self.processing_time*1000:.0f}ms', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return display_frame
            
        except queue.Empty:
            return None
    
    def start_camera(self):
        """Start camera system"""
        if self.running:
            return
        
        self.running = True
        self.camera_frame_count = 0
        self.process_frame_count = 0
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        # Start threads
        self.camera_thread_ref = threading.Thread(target=self.camera_capture_thread)
        self.process_thread_ref = threading.Thread(target=self.frame_processing_thread)
        
        self.camera_thread_ref.daemon = True
        self.process_thread_ref.daemon = True
        
        self.camera_thread_ref.start()
        self.process_thread_ref.start()
    
    def stop_camera(self):
        """Stop camera system"""
        self.running = False

def process_video_sync(video_path, latend_id, model, face_app, result_path, temp_dir, use_mask):
    """Synchronous video processing function"""
    try:
        video_swap(
            video_path, 
            latend_id, 
            model, 
            face_app, 
            result_path, 
            temp_results_dir=temp_dir,
            no_simswaplogo=False, 
            use_mask=use_mask, 
            crop_size=224
        )
        return True
    except Exception as e:
        print(f"Video processing error: {e}")
        return False

def init_model():
    """Initialize model and related components"""
    global model, face_app, transformer, transformer_Arcface, logoclass, spNorm, net
    
    # Mock command line arguments
    sys.argv = [
        'app.py',
        '--use_mask',
        '--crop_size', '224',
        '--name', 'people',
        '--Arc_path', 'arcface_model/arcface_checkpoint.tar',
        '--checkpoints_dir', './checkpoints',
        '--temp_path', './temp_results',
        '--output_path', './output/',
        '--gpu_ids', '0'
    ]
    
    opt = TestOptions().parse()
    
    # Image transformers
    transformer = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Initialize components
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    spNorm = SpecificNorm()
    
    # Load model
    torch.nn.Module.dump_patches = True
    model = create_model(opt)
    model.eval()
    
    # Initialize face detection with lower threshold
    face_app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    face_app.prepare(ctx_id=0, det_thresh=0.3, det_size=(640,640), mode='None')  # Lower detection threshold
    
    print(f"Face detection initialized with threshold: {face_app.det_thresh}")
    
    # Initialize parsing model for masks
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
    net.load_state_dict(torch.load(save_pth))
    net.eval()
    
    print("Model initialization completed")

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    init_model()
    
    # Create necessary directories
    os.makedirs("temp", exist_ok=True)
    os.makedirs("static", exist_ok=True)

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Unified Face Swap API",
        "version": "2.0.0",
        "endpoints": {
            "swap_image": "/swap_image - Image face swapping (multiple faces)",
            "swap_video": "/swap_video - Video face swapping (multiple faces)",
            "camera": "/camera - Real-time camera interface",
            "health": "/health - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/swap_image")
async def swap_image(
    source_image: UploadFile = File(..., description="Source image (provides identity)"),
    target_image: UploadFile = File(..., description="Target image (provides attributes)"),
    use_mask: bool = True
):
    """
    Whole image face swapping - supports multiple faces
    """
    try:
        task_id = str(uuid.uuid4())
        
        # Save uploaded images
        source_path = f"temp/{task_id}_source.jpg"
        target_path = f"temp/{task_id}_target.jpg"
        result_path = f"temp/{task_id}_result.jpg"
        
        # Save files
        with open(source_path, "wb") as f:
            f.write(await source_image.read())
        with open(target_path, "wb") as f:
            f.write(await target_image.read())
        
        # Perform face swapping
        with torch.no_grad():
            # Process source image with automatic pre-crop detection
            try:
                latend_id = process_source_image_for_identity(source_path, 224)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Failed to process source image: {str(e)}")
            
            # Process target image
            img_b_whole = preprocess_image_for_detection(target_path)
            
            detection_result = detect_faces_with_retry(face_app, img_b_whole, 224)
            
            if detection_result is None:
                raise HTTPException(status_code=400, detail="No face detected in target image. Please use a clear image with visible faces.")
            
            img_b_align_crop_list, b_mat_list = detection_result
            
            # Process each face
            swap_result_list = []
            b_align_crop_tenor_list = []
            
            for b_align_crop in img_b_align_crop_list:
                b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB))[None,...].cuda()
                swap_result = model(None, b_align_crop_tenor, latend_id, None, True)[0]
                swap_result_list.append(swap_result)
                b_align_crop_tenor_list.append(b_align_crop_tenor)
            
            # Reverse to whole image
            parsing_model = net if use_mask else None
            reverse2wholeimage(
                b_align_crop_tenor_list, 
                swap_result_list, 
                b_mat_list, 
                224, 
                img_b_whole, 
                logoclass, 
                result_path, 
                no_simswaplogo=False, 
                pasring_model=parsing_model, 
                use_mask=use_mask, 
                norm=spNorm
            )
        
        return FileResponse(
            result_path,
            media_type="image/jpeg",
            filename=f"swapped_{task_id}.jpg"
        )
        
    except Exception as e:
        print(f"Error in swap_image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/swap_video")
async def swap_video(
    source_image: UploadFile = File(..., description="Source image (provides identity)"),
    target_video: UploadFile = File(..., description="Target video"),
    use_mask: bool = True
):
    """
    Video face swapping - supports multiple faces
    """
    try:
        task_id = str(uuid.uuid4())
        
        source_path = f"temp/{task_id}_source.jpg"
        video_path = f"temp/{task_id}_video.mp4"
        result_path = f"temp/{task_id}_result.mp4"
        temp_dir = f"temp/{task_id}_frames"
        
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save files
        with open(source_path, "wb") as f:
            f.write(await source_image.read())
        with open(video_path, "wb") as f:
            f.write(await target_video.read())
        
        # Perform video face swapping
        with torch.no_grad():
            # Process source image with automatic pre-crop detection
            try:
                latend_id = process_source_image_for_identity(source_path, 224)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Failed to process source image: {str(e)}")
            
            # Process video asynchronously
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                executor, 
                process_video_sync, 
                video_path, latend_id, model, face_app, result_path, temp_dir, use_mask
            )
            
            if not success:
                raise HTTPException(status_code=500, detail="Video processing failed")
        
        return FileResponse(
            result_path,
            media_type="video/mp4",
            filename=f"swapped_{task_id}.mp4"
        )
        
    except Exception as e:
        print(f"Error in swap_video: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# =============================================================================
# Real-time Camera Endpoints
# =============================================================================

@app.get("/camera", response_class=HTMLResponse)
async def camera_interface():
    """Real-time camera interface"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Real-time Face Swap Camera</title>
    <meta charset="UTF-8">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            text-align: center; 
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container { 
            max-width: 800px; 
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .video-container { 
            margin: 20px 0;
            border: 3px solid #333;
            border-radius: 10px;
            overflow: hidden;
        }
        .controls { 
            margin: 20px 0;
        }
        button { 
            padding: 12px 24px; 
            margin: 8px; 
            font-size: 16px; 
            cursor: pointer;
            border: none;
            border-radius: 5px;
            color: white;
            transition: background-color 0.3s;
        }
        .upload-btn {
            background-color: #28a745;
        }
        .upload-btn:hover {
            background-color: #218838;
        }
        .start-btn {
            background-color: #007bff;
        }
        .start-btn:hover {
            background-color: #0056b3;
        }
        .stop-btn {
            background-color: #dc3545;
        }
        .stop-btn:hover {
            background-color: #c82333;
        }
        button:disabled {
            background-color: #6c757d;
            cursor: not-allowed;
        }
        #videoFeed { 
            width: 100%;
            height: auto;
            max-width: 640px;
        }
        .info { 
            margin: 10px 0; 
            color: #666;
            font-size: 14px;
        }
        .status {
            margin: 10px 0;
            padding: 10px;
            background-color: #e7f3ff;
            border-radius: 5px;
            color: #0066cc;
        }
        .upload-section {
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .file-input {
            margin: 10px;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .error {
            color: #dc3545;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .workflow {
            margin: 20px 0;
            padding: 15px;
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
        }
        .workflow h3 {
            color: #856404;
            margin-top: 0;
        }
        .workflow ol {
            text-align: left;
            display: inline-block;
            color: #856404;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Real-time Face Swap Camera</h1>
        
        <div class="workflow">
            <h3>📋 Workflow</h3>
            <ol>
                <li><strong>Upload</strong> source image (the face you want to swap TO)</li>
                <li><strong>Start</strong> camera to begin real-time face swapping</li>
                <li><strong>Stop</strong> camera when done</li>
                <li>To change face: <strong>Stop</strong> → <strong>Upload new image</strong> → <strong>Start</strong></li>
            </ol>
        </div>
        
        <div class="upload-section">
            <h3>Step 1: Upload Source Image</h3>
            <p style="color: #666; font-size: 14px;">Choose a clear image with a visible face. The system will detect and extract the face identity.</p>
            <input type="file" id="sourceImage" accept="image/*" class="file-input">
            <button id="uploadBtn" class="upload-btn" onclick="uploadSource()">Upload & Initialize</button>
        </div>
        
        <div class="video-container">
            <img id="videoFeed" src="" alt="Camera feed will appear here after uploading source image">
        </div>
        
        <div class="controls">
            <button id="startBtn" class="start-btn" onclick="startCamera()" disabled>Start Camera</button>
            <button id="stopBtn" class="stop-btn" onclick="stopCamera()" disabled>Stop Camera</button>
        </div>
        
        <div class="status" id="statusDiv">
            <p><strong>Status:</strong> <span id="statusText">Ready - Please upload a source image</span></p>
        </div>
        
        <div class="info">
            <p><strong>Features:</strong></p>
            <ul style="text-align: left; display: inline-block;">
                <li>Real-time face swapping with multi-threading</li>
                <li>Automatic face detection and alignment</li>
                <li>High-quality face parsing and blending</li>
                <li>Performance monitoring (FPS and processing time)</li>
            </ul>
            <p style="color: #666; font-size: 12px;">
                <strong>Tips:</strong> Use well-lit, clear images with visible faces for best results. The first face detected will be used as the identity source.
            </p>
        </div>
    </div>

    <script>
        let currentSession = null;
        let isStarted = false;
        
        const uploadBtn = document.getElementById('uploadBtn');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const sourceImageInput = document.getElementById('sourceImage');
        
        function updateStatus(message, isError = false) {
            const statusText = document.getElementById('statusText');
            const statusDiv = document.getElementById('statusDiv');
            
            statusText.textContent = message;
            
            if (isError) {
                statusDiv.className = 'error';
            } else {
                statusDiv.className = 'status';
            }
        }
        
        function updateButtonStates() {
            if (currentSession && !isStarted) {
                // Session exists, camera not started
                startBtn.disabled = false;
                stopBtn.disabled = true;
                uploadBtn.disabled = false;
                sourceImageInput.disabled = false;
            } else if (currentSession && isStarted) {
                // Session exists, camera started
                startBtn.disabled = true;
                stopBtn.disabled = false;
                uploadBtn.disabled = true;
                sourceImageInput.disabled = true;
            } else {
                // No session
                startBtn.disabled = true;
                stopBtn.disabled = true;
                uploadBtn.disabled = false;
                sourceImageInput.disabled = false;
            }
        }
        
        async function uploadSource() {
            const file = sourceImageInput.files[0];
            
            if (!file) {
                updateStatus('Please select an image file', true);
                return;
            }
            
            updateStatus('Uploading and processing source image... This may take a few moments.');
            uploadBtn.disabled = true;
            
            const formData = new FormData();
            formData.append('source_image', file);
            
            try {
                const response = await fetch('/camera/upload_source', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    currentSession = data.session_id;
                    isStarted = false;
                    updateStatus('✅ Source image processed successfully! You can now start the camera.');
                    document.getElementById('videoFeed').src = `/camera/feed/${currentSession}`;
                    updateButtonStates();
                } else {
                    updateStatus('❌ Error: ' + data.detail, true);
                    uploadBtn.disabled = false;
                }
            } catch (error) {
                updateStatus('❌ Upload failed: ' + error.message, true);
                uploadBtn.disabled = false;
            }
        }
        
        async function startCamera() {
            if (!currentSession) {
                updateStatus('Please upload a source image first', true);
                return;
            }
            
            updateStatus('🎥 Starting camera...');
            
            try {
                const response = await fetch(`/camera/start/${currentSession}`, {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    isStarted = true;
                    updateStatus('🎉 Camera started - Face swapping in real-time!');
                    updateButtonStates();
                } else {
                    updateStatus('❌ Error: ' + data.detail, true);
                }
            } catch (error) {
                updateStatus('❌ Start failed: ' + error.message, true);
            }
        }
        
        async function stopCamera() {
            if (!currentSession) return;
            
            updateStatus('⏹️ Stopping camera...');
            
            try {
                const response = await fetch(`/camera/stop/${currentSession}`, {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    isStarted = false;
                    updateStatus('⏹️ Camera stopped - You can upload a new image or restart');
                    updateButtonStates();
                } else {
                    updateStatus('❌ Error: ' + data.detail, true);
                }
            } catch (error) {
                updateStatus('❌ Stop failed: ' + error.message, true);
            }
        }
        
        // Initialize button states
        updateButtonStates();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.post("/camera/upload_source")
async def upload_source_image(source_image: UploadFile = File(...)):
    """Upload source image for real-time face swap"""
    try:
        session_id = str(uuid.uuid4())
        source_path = f"temp/{session_id}_source.jpg"
        
        # Save source image
        with open(source_path, "wb") as f:
            f.write(await source_image.read())
        
        # Create camera instance
        camera_instance = RealtimeFaceSwapCamera(
            source_image_path=source_path,
            crop_size=224,
            use_mask=True,
            camera_id=0
        )
        
        camera_instances[session_id] = camera_instance
        
        return {"session_id": session_id, "status": "uploaded"}
        
    except Exception as e:
        print(f"Error in upload_source_image: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/camera/start/{session_id}")
async def start_camera_session(session_id: str):
    """Start camera for a session"""
    if session_id not in camera_instances:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        camera_instances[session_id].start_camera()
        return {"status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Start failed: {str(e)}")

@app.post("/camera/stop/{session_id}")
async def stop_camera_session(session_id: str):
    """Stop camera for a session"""
    if session_id not in camera_instances:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        camera_instances[session_id].stop_camera()
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stop failed: {str(e)}")

@app.get("/camera/feed/{session_id}")
async def camera_feed(session_id: str):
    """Video feed for a session"""
    if session_id not in camera_instances:
        raise HTTPException(status_code=404, detail="Session not found")
    
    def generate():
        camera_instance = camera_instances[session_id]
        while True:
            frame = camera_instance.get_latest_frame()
            if frame is not None:
                try:
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"Error encoding frame: {e}")
            time.sleep(0.03)  # ~30 FPS
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)