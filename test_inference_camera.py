'''
Real-time face swap using camera input with web interface (Fixed)
'''

import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.reverse2original import reverse2wholeimage
import os
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
import argparse
import time
import sys
import threading
from flask import Flask, render_template, Response, request, jsonify
import tempfile

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

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
    else:
        pass

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
    
    # Return the processed image instead of saving
    return final_img

class RealtimeFaceSwapWeb:
    def __init__(self, source_image_path, crop_size=224, use_mask=True):
        self.crop_size = crop_size
        self.use_mask = use_mask
        self.source_image_path = source_image_path
        
        # Control variables
        self.show_original = False
        self.running = False
        self.frame_count = 0
        self.start_time = time.time()
        self.current_frame = None
        self.processing_time = 0
        self.fps = 0
        
        # Initialize models
        self.setup_models()
        self.setup_source_identity(source_image_path)
        
    def setup_models(self):
        """Initialize all models"""
        print("Loading models...")
        
        # Create options
        original_argv = sys.argv.copy()
        try:
            sys.argv = ['realtime_camera_swap_web_fixed.py']
            opt = TestOptions().parse(save=False)
            
            opt.crop_size = self.crop_size
            opt.use_mask = self.use_mask
            opt.no_simswaplogo = False
            opt.name = 'people'
            opt.Arc_path = 'arcface_model/arcface_checkpoint.tar'
            opt.isTrain = False
            
        finally:
            sys.argv = original_argv
        
        self.opt = opt
        
        # Set mode
        if self.crop_size == 512:
            self.opt.which_epoch = 550000
            self.opt.name = '512'
            mode = 'ffhq'
        else:
            mode = 'None'
            
        # Load models
        self.logoclass = watermark_image('./simswaplogo/simswaplogo.png')
        self.model = create_model(self.opt)
        self.model.eval()
        self.spNorm = SpecificNorm()
        
        self.app = Face_detect_crop(name='antelope', root='./insightface_func/models')
        self.app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640), mode=mode)
        
        if self.use_mask:
            print("Loading face parsing model...")
            n_classes = 19
            self.net = BiSeNet(n_classes=n_classes)
            self.net.cuda()
            save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
            self.net.load_state_dict(torch.load(save_pth))
            self.net.eval()
        else:
            self.net = None
            
        print("Models loaded successfully!")
    
    def setup_source_identity(self, source_image_path):
        """Process source image to extract identity features"""
        print(f"Processing source image: {source_image_path}")
        
        with torch.no_grad():
            img_a_whole = cv2.imread(source_image_path)
            if img_a_whole is None:
                raise ValueError(f"Could not load source image: {source_image_path}")
            
            img_a_align_crop, _ = self.app.get(img_a_whole, self.crop_size)
            if len(img_a_align_crop) == 0:
                raise ValueError("No face detected in source image")
            
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
            img_a = transformer_Arcface(img_a_align_crop_pil)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
            
            img_id = img_id.cuda()
            img_id_downsample = F.interpolate(img_id, size=(112,112))
            self.latend_id = self.model.netArc(img_id_downsample)
            self.latend_id = F.normalize(self.latend_id, p=2, dim=1)
            
        print("Source identity extracted!")
    
    def process_frame(self, frame):
        """Process a single frame for face swap"""
        try:
            with torch.no_grad():
                img_b_align_crop_list, b_mat_list = self.app.get(frame, self.crop_size)
                
                if len(img_b_align_crop_list) == 0:
                    return frame
                
                swap_result_list = []
                b_align_crop_tenor_list = []
                
                for b_align_crop in img_b_align_crop_list:
                    b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB))[None,...].cuda()
                    swap_result = self.model(None, b_align_crop_tenor, self.latend_id, None, True)[0]
                    swap_result_list.append(swap_result)
                    b_align_crop_tenor_list.append(b_align_crop_tenor)
                
                # Use the modified function that returns the image instead of saving
                output_frame = reverse2wholeimage_realtime(
                    b_align_crop_tenor_list, swap_result_list, b_mat_list, 
                    self.crop_size, frame, self.logoclass, 
                    self.opt.no_simswaplogo, pasring_model=self.net, 
                    use_mask=self.use_mask, norm=self.spNorm
                )
                
                return output_frame
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame
    
    def camera_thread(self, camera_id=0):
        """Camera capture thread"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Could not open camera {camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera thread started!")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Mirror frame
            frame = cv2.flip(frame, 1)
            
            # Process frame
            start_time = time.time()
            if self.show_original:
                result_frame = frame.copy()
            else:
                result_frame = self.process_frame(frame)
            self.processing_time = time.time() - start_time
            
            # Update FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            self.fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Add text overlay
            status_text = "Original" if self.show_original else "Face Swapped"
            cv2.putText(result_frame, f'Status: {status_text}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_frame, f'FPS: {self.fps:.1f}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_frame, f'Process: {self.processing_time*1000:.1f}ms', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            self.current_frame = result_frame
            
            time.sleep(0.03)  # Small delay to prevent too high CPU usage
        
        cap.release()
        print("Camera thread stopped!")

# Flask app
app = Flask(__name__)
face_swap_instance = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global face_swap_instance
    if face_swap_instance is not None:
        face_swap_instance.running = True
        camera_thread = threading.Thread(target=face_swap_instance.camera_thread, args=(0,))
        camera_thread.daemon = True
        camera_thread.start()
        return jsonify({'status': 'started'})
    return jsonify({'status': 'error', 'message': 'Face swap not initialized'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global face_swap_instance
    if face_swap_instance is not None:
        face_swap_instance.running = False
        return jsonify({'status': 'stopped'})
    return jsonify({'status': 'error'})

@app.route('/toggle_view', methods=['POST'])
def toggle_view():
    global face_swap_instance
    if face_swap_instance is not None:
        face_swap_instance.show_original = not face_swap_instance.show_original
        return jsonify({'status': 'toggled', 'show_original': face_swap_instance.show_original})
    return jsonify({'status': 'error'})

@app.route('/save_frame', methods=['POST'])
def save_frame():
    global face_swap_instance
    if face_swap_instance is not None and face_swap_instance.current_frame is not None:
        # Create a unique filename
        timestamp = int(time.time())
        filename = f'capture_{timestamp}.jpg'
        
        # Save the frame
        try:
            cv2.imwrite(filename, face_swap_instance.current_frame)
            return jsonify({'status': 'saved', 'filename': filename})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    return jsonify({'status': 'error'})

@app.route('/video_feed')
def video_feed():
    def generate():
        global face_swap_instance
        while True:
            if face_swap_instance is not None and face_swap_instance.current_frame is not None:
                try:
                    ret, buffer = cv2.imencode('.jpg', face_swap_instance.current_frame)
                    if ret:
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    print(f"Error encoding frame: {e}")
            time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    global face_swap_instance
    
    parser = argparse.ArgumentParser(description='Real-time face swap using camera with web interface')
    parser.add_argument('--source', type=str, required=True, help='Path to source image')
    parser.add_argument('--crop_size', type=int, default=224, help='Crop size')
    parser.add_argument('--use_mask', action='store_true', default=True, help='Use mask')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source):
        print(f"Source image not found: {args.source}")
        return
    
    try:
        # Initialize face swap
        face_swap_instance = RealtimeFaceSwapWeb(
            source_image_path=args.source,
            crop_size=args.crop_size,
            use_mask=args.use_mask
        )
        
        # Create templates directory
        os.makedirs('templates', exist_ok=True)
        
        # Create HTML template
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Real-time Face Swap</title>
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
            background-color: #007bff;
            color: white;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #0056b3;
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
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Real-time Face Swap</h1>
        
        <div class="info">
            <p><strong>Source Image:</strong> """ + args.source + """</p>
        </div>
        
        <div class="video-container">
            <img id="videoFeed" src="{{ url_for('video_feed') }}" alt="Video Feed">
        </div>
        
        <div class="controls">
            <button onclick="startCamera()">Start Camera</button>
            <button onclick="stopCamera()">Stop Camera</button>
            <button onclick="toggleView()">Toggle View</button>
            <button onclick="saveFrame()">Save Frame</button>
        </div>
        
        <div class="status">
            <p><strong>Instructions:</strong></p>
            <ul style="text-align: left; display: inline-block;">
                <li>Click "Start Camera" to begin face swapping</li>
                <li>Click "Toggle View" to switch between original and swapped</li>
                <li>Click "Save Frame" to save the current frame</li>
                <li>Click "Stop Camera" to stop the camera</li>
            </ul>
        </div>
    </div>

    <script>
        let isStarted = false;
        
        function startCamera() {
            if (!isStarted) {
                fetch('/start_camera', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        console.log('Camera started:', data);
                        isStarted = true;
                    })
                    .catch(error => console.error('Error:', error));
            }
        }
        
        function stopCamera() {
            if (isStarted) {
                fetch('/stop_camera', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        console.log('Camera stopped:', data);
                        isStarted = false;
                    })
                    .catch(error => console.error('Error:', error));
            }
        }
        
        function toggleView() {
            fetch('/toggle_view', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    console.log('View toggled:', data);
                })
                .catch(error => console.error('Error:', error));
        }
        
        function saveFrame() {
            fetch('/save_frame', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'saved') {
                        alert('Frame saved as: ' + data.filename);
                    } else {
                        alert('Error saving frame: ' + data.message);
                    }
                })
                .catch(error => console.error('Error:', error));
        }
    </script>
</body>
</html>"""
        
        # Write HTML file with UTF-8 encoding
        with open('templates/index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Starting web server on http://localhost:{args.port}")
        print("Open your browser and go to the URL above")
        print("Press Ctrl+C to stop the server")
        
        app.run(host='0.0.0.0', port=args.port, debug=False)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()