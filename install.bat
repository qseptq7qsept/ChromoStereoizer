@echo off
REM Clone the repository
git clone https://github.com/DepthAnything/Depth-Anything-V2
if errorlevel 1 goto error

cd Depth-Anything-V2

REM Create a Python virtual environment called "env"
python -m venv env
if errorlevel 1 goto error

REM Activate the virtual environment
call env\Scripts\activate
if errorlevel 1 goto error

REM Install packages from requirements.txt within the virtual environment
pip install -r requirements.txt
if errorlevel 1 goto error

REM Install additional dependencies needed for the code
pip install torch numpy pillow PySide6 transformers
if errorlevel 1 goto error

REM Create the folder for the model file: depth_anything_v2\Depth-Anything-V2-Small-hf
mkdir depth_anything_v2\Depth-Anything-V2-Small-hf
if errorlevel 1 goto error

REM Download the model file from Hugging Face and save it in the new folder
curl -L "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true" -o depth_anything_v2\Depth-Anything-V2-Small-hf\depth_anything_v2_vits.pth
if errorlevel 1 goto error

REM Prompt the installation complete message
echo Installation complete! -q7
pause
goto end

:error
echo An error occurred during installation.
pause

:end
