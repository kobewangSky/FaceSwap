# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
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

app = FastAPI(title="Face Swap API", version="1.0.0")

# Global variables
model = None
face_app = None
transformer = None
transformer_Arcface = None
logoclass = None
spNorm = None
net = None

# Thread pool executor for concurrent processing
executor = ThreadPoolExecutor(max_workers=4)  # Limit concurrent threads

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

# Task queue
task_queue = asyncio.Queue()
task_status = {}

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

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
    
    # Initialize face detection (using multi version)
    face_app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    face_app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640), mode='None')
    
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

@app.post("/swap_image")
async def swap_image(
    source_image: UploadFile = File(..., description="Source image (provides identity)"),
    target_image: UploadFile = File(..., description="Target image (provides attributes)"),
    use_mask: bool = True
):
    """
    Whole image face swapping - supports multiple faces
    - source_image: Source image providing identity features
    - target_image: Target image where all faces will be replaced
    - use_mask: Whether to use face parsing mask for better blending
    """
    try:
        task_id = str(uuid.uuid4())
        
        # Save uploaded images
        source_path = f"temp/{task_id}_source.jpg"
        target_path = f"temp/{task_id}_target.jpg"
        result_path = f"temp/{task_id}_result.jpg"
        
        # Ensure temp directory exists
        os.makedirs("temp", exist_ok=True)
        
        # Save files
        with open(source_path, "wb") as f:
            f.write(await source_image.read())
        with open(target_path, "wb") as f:
            f.write(await target_image.read())
        
        # Perform face swapping
        with torch.no_grad():
            # Process source image (identity)
            img_a_whole = cv2.imread(source_path)
            img_a_align_crop_list, _ = face_app.get(img_a_whole, 224)
            
            if img_a_align_crop_list is None:
                raise HTTPException(status_code=400, detail="No face detected in source image")
            
            # Use first face for identity
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop_list[0], cv2.COLOR_BGR2RGB))
            img_a = transformer_Arcface(img_a_align_crop_pil)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
            
            # Convert to CUDA tensor
            img_id = img_id.cuda()
            
            # Create identity features
            img_id_downsample = F.interpolate(img_id, size=(112,112))
            latend_id = model.netArc(img_id_downsample)
            latend_id = F.normalize(latend_id, p=2, dim=1)
            
            # Process target image (multiple faces)
            img_b_whole = cv2.imread(target_path)
            img_b_align_crop_list, b_mat_list = face_app.get(img_b_whole, 224)
            
            if img_b_align_crop_list is None:
                raise HTTPException(status_code=400, detail="No face detected in target image")
            
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
        
        # Return result file
        return FileResponse(
            result_path,
            media_type="image/jpeg",
            filename=f"swapped_{task_id}.jpg"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/swap_video")
async def swap_video(
    source_image: UploadFile = File(..., description="Source image (provides identity)"),
    target_video: UploadFile = File(..., description="Target video"),
    use_mask: bool = True
):
    """
    Video face swapping - supports multiple faces
    - source_image: Source image providing identity features
    - target_video: Target video where faces will be replaced with source image's face
    - use_mask: Whether to use face parsing mask for better blending
    """
    try:
        # Generate unique ID
        task_id = str(uuid.uuid4())
        
        # Save uploaded files
        source_path = f"temp/{task_id}_source.jpg"
        video_path = f"temp/{task_id}_video.mp4"
        result_path = f"temp/{task_id}_result.mp4"
        temp_dir = f"temp/{task_id}_frames"
        
        # Ensure directories exist
        os.makedirs("temp", exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save files
        with open(source_path, "wb") as f:
            f.write(await source_image.read())
        with open(video_path, "wb") as f:
            f.write(await target_video.read())
        
        # Perform video face swapping
        with torch.no_grad():
            # Process source image
            img_a_whole = cv2.imread(source_path)
            img_a_align_crop_list, _ = face_app.get(img_a_whole, 224)
            
            if img_a_align_crop_list is None:
                raise HTTPException(status_code=400, detail="No face detected in source image")
            
            # Use first face for identity
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop_list[0], cv2.COLOR_BGR2RGB))
            img_a = transformer_Arcface(img_a_align_crop_pil)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
            
            # Convert to CUDA tensor
            img_id = img_id.cuda()
            
            # Create identity features
            img_id_downsample = F.interpolate(img_id, size=(112,112))
            latend_id = model.netArc(img_id_downsample)
            latend_id = F.normalize(latend_id, p=2, dim=1)
            
            # Process video asynchronously
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                executor, 
                process_video_sync, 
                video_path, latend_id, model, face_app, result_path, temp_dir, use_mask
            )
            
            if not success:
                raise HTTPException(status_code=500, detail="Video processing failed")
        
        # Return result video
        return FileResponse(
            result_path,
            media_type="video/mp4",
            filename=f"swapped_{task_id}.mp4"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Face Swap API",
        "version": "1.0.0",
        "endpoints": {
            "swap_image": "/swap_image - Whole image face swapping (multiple faces)",
            "swap_video": "/swap_video - Video face swapping (multiple faces)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)