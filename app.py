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
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.videoswap import video_swap
import os
import tempfile
import uuid
from typing import Optional
import sys

app = FastAPI(title="Face Swap API", version="1.0.0")

# Global variables
model = None
face_app = None
transformer = None
transformer_Arcface = None

def init_model():
    """Initialize model and related components"""
    global model, face_app, transformer, transformer_Arcface
    
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
    
    # Load model
    torch.nn.Module.dump_patches = True
    model = create_model(opt)
    model.eval()
    
    # Initialize face detection
    face_app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    face_app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640), mode='None')
    
    print("Model initialization completed")

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    init_model()

@app.post("/swap_image")
async def swap_image(
    source_image: UploadFile = File(..., description="Source image (provides identity)"),
    target_image: UploadFile = File(..., description="Target image (provides attributes)")
):
    """
    Image face swapping
    - source_image: Source image providing identity features
    - target_image: Target image providing pose, expression, etc.
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
            img_a = Image.open(source_path).convert('RGB')
            img_a = transformer_Arcface(img_a)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
            
            # Process target image (attributes)
            img_b = Image.open(target_path).convert('RGB')
            img_b = transformer(img_b)
            img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])
            
            # Convert to CUDA tensors
            img_id = img_id.cuda()
            img_att = img_att.cuda()
            
            # Create identity features
            img_id_downsample = F.interpolate(img_id, size=(112,112))
            latend_id = model.netArc(img_id_downsample)
            latend_id = latend_id.detach().to('cpu')
            latend_id = latend_id/np.linalg.norm(latend_id,axis=1,keepdims=True)
            latend_id = latend_id.to('cuda')
            
            # Generate swapped result
            img_fake = model(img_id, img_att, latend_id, latend_id, True)
            
            # Process output
            full = img_fake[0].detach()
            full = full.permute(1, 2, 0)
            output = full.to('cpu')
            output = np.array(output)
            output = output[..., ::-1]  # RGB to BGR
            output = output * 255
            
            # Save result
            cv2.imwrite(result_path, output)
        
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
    target_video: UploadFile = File(..., description="Target video")
):
    """
    Video face swapping
    - source_image: Source image providing identity features
    - target_video: Target video where faces will be replaced with source image's face
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
            img_a_align_crop, _ = face_app.get(img_a_whole, 224)
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
            img_a = transformer_Arcface(img_a_align_crop_pil)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
            
            # Convert to CUDA tensor
            img_id = img_id.cuda()
            
            # Create identity features
            img_id_downsample = F.interpolate(img_id, size=(112,112))
            latend_id = model.netArc(img_id_downsample)
            latend_id = F.normalize(latend_id, p=2, dim=1)
            
            # Process video
            video_swap(
                video_path, 
                latend_id, 
                model, 
                face_app, 
                result_path, 
                temp_results_dir=temp_dir,
                no_simswaplogo=False, 
                use_mask=True, 
                crop_size=224
            )
        
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
            "swap_image": "/swap_image - Image face swapping",
            "swap_video": "/swap_video - Video face swapping"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)