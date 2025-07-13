# Face Swap Deep Learning Project

A comprehensive face swapping system using deep learning with training pipeline, production API, and dataset management tools.

**This project is adapted and enhanced from [SimSwap](https://github.com/neuralchen/SimSwap) with additional features including production API, dataset management tools, and improved training pipeline.**

## 📋 Features

- **Training Pipeline**: Complete training script with tensorboard visualization
- **Production API**: FastAPI-based REST API with multi-threading support
- **Dataset Management**: Tools for dataset updates, validation, and duplicate detection
- **Video Processing**: Face swapping on video frames with temporal consistency
- **Multi-face Support**: Handle multiple faces in single image/video

## 📊 Dataset

This project uses the **LFW (Labeled Faces in the Wild)** dataset structure:

- **Location**: `./training_data/`
- **Format**: Organized by person names (e.g., `./training_data/Person_Name/`)
- **Structure**: Each person folder contains their face images
- **Supported formats**: JPG, JPEG, PNG

### Current Dataset Stats
The `training_data` folder contains multiple person directories with face images organized by individual names.

## 🛠 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- PyTorch, OpenCV, FastAPI

### Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd FaceSwap

# Create conda environment
conda create -n faceswap python=3.8
conda activate faceswap

# Install other dependencies
pip install -r requirements.txt
```

### Required Model Files

Download and place the following model files in their respective directories:

```
FaceSwap/
├── arcface_model/
│   └── arcface_checkpoint.tar          # ArcFace identity model
├── insightface_func/
│   └── models/
│       └── antelope/                   # InsightFace detection models
│           ├── 1k3d68.onnx
│           ├── 2d106det.onnx
│           ├── genderage.onnx
│           └── scrfd_10g_bnkps.onnx
```


## 🎯 Usage

### 1. Dataset Management

Use `data_toolkit.py` to manage your training data:

```bash
# Add a new person to dataset
python data_toolkit.py add --person john_doe --images ./john_images --dataset ./training_data

# Merge two datasets
python data_toolkit.py merge --source ./other_dataset --dataset ./training_data

# Validate dataset (check for face detection, duplicates)
python data_toolkit.py validate --dataset ./training_data

# Remove duplicates from dataset
python data_toolkit.py clean --dataset ./training_data

# List all people in dataset
python data_toolkit.py list --dataset ./training_data

# Delete a person from dataset
python data_toolkit.py delete --person john_doe --dataset ./training_data

# Show dataset statistics
python data_toolkit.py stats --dataset ./training_data
```

**Features:**
- Face detection validation (ensures each image has at least one face)
- Option to handle multiple faces (uses first/largest face)
- Duplicate detection using MD5 hash
- Dataset statistics and validation
- Safe person deletion with confirmation

### 2. Training

Use `train.py` to train the face swap model:

```bash
# Basic training (224x224 resolution)
python train.py --name my_experiment --dataset ./training_data --batchSize 8 --gpu_ids 0 --Gdeep False

# Training with tensorboard
python train.py --name my_experiment --dataset ./training_data --batchSize 8 --gpu_ids 0 --use_tensorboard True --Gdeep False

# Continue training from checkpoint
python train.py --name my_experiment --continue_train True --which_epoch 10000 --Gdeep False

# Training with custom ArcFace path
python train.py --name my_experiment --dataset ./training_data --Arc_path arcface_model/arcface_checkpoint.tar --Gdeep False

# High resolution training (512x512)
python train.py --name my_experiment_512 --dataset ./training_data --batchSize 4 --gpu_ids 0 --Gdeep True
```

**Training Parameters:**
- `--name`: Experiment name (saves to `./checkpoints/[name]/`)
- `--dataset`: Path to training data
- `--batchSize`: Batch size (default: 4)
- `--gpu_ids`: GPU ID to use
- `--use_tensorboard`: Enable tensorboard logging
- `--Arc_path`: Path to ArcFace model (default: `arcface_model/arcface_checkpoint.tar`)
- `--Gdeep`: Generator network depth (default: False)
  - **False**: Shallow network for 224x224 training (recommended)
    - Faster training and less GPU memory usage
    - Suitable for most face swapping tasks
    - More stable training convergence
  - **True**: Deep network for 512x512 high-resolution training
    - Better detail preservation
    - Requires more GPU memory (>10GB)
    - Longer training time
- `--niter`: Number of training iterations
- `--lr`: Learning rate (default: 0.0004)

**Tensorboard Visualization:**
```bash
# View training progress
tensorboard --logdir ./checkpoints/[experiment_name]/summary
```

**Training Metrics:**
- `G_Loss`: Generator adversarial loss
- `G_ID`: Identity preservation loss
- `G_Rec`: Reconstruction loss
- `G_feat_match`: Feature matching loss
- `D_fake`: Discriminator loss on fake images
- `D_real`: Discriminator loss on real images
- `D_loss`: Total discriminator loss

### 3. Production API

Use `app.py` to start the production API server:

```bash
# Start API server
python app.py
```

**API Endpoints:**

#### Image Face Swap
```bash
curl -X POST "http://localhost:8000/swap_image" \
  -F "source_image=@source.jpg" \
  -F "target_image=@target.jpg" \
  -F "use_mask=true" \
  --output result.jpg
```

#### Video Face Swap
```bash
curl -X POST "http://localhost:8000/swap_video" \
  -F "source_image=@source.jpg" \
  -F "target_video=@video.mp4" \
  -F "use_mask=true" \
  --output result.mp4
```

#### Health Check
```bash
curl http://localhost:8000/health
```

**API Features:**
- Multi-threading support (4 concurrent workers)
- Automatic face detection and cropping
- Face parsing mask for better blending
- Support for multiple faces in single image/video

### 4. Real-time Camera Face Swap

Use `test_inference_camera.py` for real-time face swapping with webcam:

```bash
# Start real-time camera face swap
python test_inference_camera.py --source ./demo_file/Iron_man.jpg --crop_size 224 --use_mask --port 5000
```

**Parameters:**
- `--source`: Path to source image (provides face identity)
- `--crop_size`: Input image size (default: 224)
- `--use_mask`: Use face parsing mask for better blending
- `--port`: Web server port (default: 5000)

**Features:**
- **Multi-threaded Architecture**: 
  - **Camera Thread**: High-FPS camera capture (~30 FPS)
  - **Processing Thread**: Face swap computation (~1-2 FPS)
  - **Display Thread**: Web interface updates (~30 FPS)
- **Web Interface**: Real-time video feed with controls
- **Performance Monitoring**: Live FPS and processing time display
- **Interactive Controls**:
  - Start/Stop camera
  - Toggle between original and swapped view
  - Save current frame
- **Optimized Performance**:
  - Separate threads prevent blocking
  - Frame queues for smooth playback
  - Efficient memory management

**Web Interface:**
1. Open browser and go to `http://localhost:5000`
2. Click "Start Camera" to begin face swapping
3. Use "Toggle View" to switch between original and swapped
4. Click "Save Frame" to capture the current frame
5. Monitor real-time performance metrics

**Performance Metrics:**
- **Camera FPS**: Real-time camera capture rate
- **Process FPS**: Face swap processing rate  
- **Process Time**: Time per face swap operation
- **Memory Usage**: GPU memory consumption


### 5. API Usage

#### Using curl commands

```bash
# Basic image face swap
curl -X POST "http://localhost:8000/swap_image" \
  -F "source_image=@source.jpg" \
  -F "target_image=@target.jpg" \
  -F "use_mask=true" \
  --output result.jpg

# Image face swap without mask
curl -X POST "http://localhost:8000/swap_image" \
  -F "source_image=@source.jpg" \
  -F "target_image=@target.jpg" \
  -F "use_mask=false" \
  --output result.jpg

# Video face swap
curl -X POST "http://localhost:8000/swap_video" \
  -F "source_image=@source.jpg" \
  -F "target_video=@video.mp4" \
  -F "use_mask=true" \
  --output result.mp4

# Video face swap without mask
curl -X POST "http://localhost:8000/swap_video" \
  -F "source_image=@source.jpg" \
  -F "target_video=@video.mp4" \
  -F "use_mask=false" \
  --output result.mp4

# Health check
curl http://localhost:8000/health

# Get API information
curl http://localhost:8000/

# Example with demo files
curl -X POST "http://localhost:8000/swap_image" \
  -F "source_image=@./demo_file/Iron_man.jpg" \
  -F "target_image=@./demo_file/multi_people.jpg" \
  -F "use_mask=true" \
  --output ./output/result_swap.jpg

curl -X POST "http://localhost:8000/swap_video" \
  -F "source_image=@./demo_file/Iron_man.jpg" \
  -F "target_video=@./demo_file/short_clip.mp4" \
  -F "use_mask=true" \
  --output ./output/result_video.mp4
```

## 📁 Project Structure

```
FaceSwap/
├── training_data/              # Training dataset (LFW format)
│   ├── Person_Name_1/         # Individual person folders
│   ├── Person_Name_2/
│   └── ...
├── checkpoints/               # Training checkpoints and models
├── temp/                      # Temporary files for API
├── output/                    # Output results
├── train.py                   # Training script
├── app.py                     # Production API server
├── data_toolkit.py            # Dataset management tool
├── models/                    # Model definitions
├── util/                      # Utility functions
├── insightface_func/          # Face detection functions
│   └── models/
│       └── antelope/          # InsightFace detection models
├── parsing_model/             # Face parsing model
│   └── checkpoint/
│       └── 79999_iter.pth     # Face parsing checkpoint
├── arcface_model/             # ArcFace identity model
│   └── arcface_checkpoint.tar # ArcFace model checkpoint
└── simswaplogo/               # Logo for watermarking
    └── simswaplogo.png        # Logo file
```

## 🚀 Quick Start

```bash
# 1. Download required models (see Required Model Files section)

# 2. Prepare dataset
python data_toolkit.py validate --dataset ./training_data

# 3. Train model
python train.py --name my_model --dataset ./training_data --batchSize 8 --use_tensorboard True

# 4. Monitor training
tensorboard --logdir ./checkpoints/my_model/summary

# 5. Start API server
python app.py

# 6. Test API
curl -X POST "http://localhost:8000/swap_image" \
  -F "source_image=@demo_file/source.jpg" \
  -F "target_image=@demo_file/target.jpg" \
  --output result.jpg
```


## 🔍 Monitoring

### Tensorboard Metrics
- Loss curves (Generator, Discriminator, Identity, Reconstruction)
- Feature matching loss progression


## 🙏 Acknowledgments

This project is based on [SimSwap](https://github.com/neuralchen/SimSwap) by Xuanhong Chen et al. We thank the original authors for their excellent work on face swapping technology.


## 📄 License

This project is for educational and research purposes only. Please use responsibly and respect privacy rights.

## 🆘 Support

For issues and questions:
- Check the training logs in `./checkpoints/[experiment_name]/`
- Use `python data_toolkit.py validate` to check dataset integrity
- Monitor API health with `curl http://localhost:8000/health`

---

**Note**: Ensure you have proper permissions and consent when working with facial data.
