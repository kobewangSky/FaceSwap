@echo off
echo Installing PyTorch GPU version...
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121

echo Verifying PyTorch installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo Installing other dependencies...
pip install -r requirements.txt

echo Installation complete!
pause