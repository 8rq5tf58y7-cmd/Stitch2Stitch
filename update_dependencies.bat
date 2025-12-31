@echo off
REM Update Stitch2Stitch dependencies
REM Run this to install missing packages like pycolmap

echo ========================================
echo Stitch2Stitch Dependency Updater
echo ========================================
echo.

cd /d "%~dp0"

REM Find and activate venv
if exist "venv\Scripts\activate.bat" (
    echo Found virtual environment in project directory.
    call venv\Scripts\activate.bat
) else if exist "%LOCALAPPDATA%\Stitch2Stitch\venv\Scripts\activate.bat" (
    echo Found virtual environment in install directory.
    call "%LOCALAPPDATA%\Stitch2Stitch\venv\Scripts\activate.bat"
) else (
    echo ERROR: Virtual environment not found!
    echo Please run the installer first.
    pause
    exit /b 1
)

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo ========================================
echo Installing/Updating dependencies...
echo ========================================
echo.

REM Install from requirements.txt
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo requirements.txt not found, installing key packages manually...
)

echo.
echo ========================================
echo Installing PyCOLMAP (COLMAP Python bindings)...
echo ========================================
pip install pycolmap

echo.
echo ========================================
echo Checking for NVIDIA GPU for PyTorch CUDA...
echo ========================================

REM Check for NVIDIA GPU
set HAS_NVIDIA=0
for /f "tokens=*" %%i in ('wmic path win32_VideoController get name ^| findstr /i "NVIDIA"') do set HAS_NVIDIA=1

if %HAS_NVIDIA%==1 (
    echo NVIDIA GPU detected! Installing PyTorch with CUDA support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else (
    echo No NVIDIA GPU detected. Using CPU version of PyTorch.
    pip install torch torchvision
)

echo.
echo ========================================
echo Verifying installations...
echo ========================================
echo.

echo Checking pycolmap...
python -c "import pycolmap; print(f'  pycolmap version: {pycolmap.__version__}')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   ERROR: pycolmap not installed correctly!
) else (
    echo   pycolmap OK
)

echo.
echo Checking PyTorch...
python -c "import torch; print(f'  PyTorch version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   ERROR: PyTorch not installed correctly!
) else (
    echo   PyTorch OK
)

echo.
echo ========================================
echo Update complete!
echo ========================================
echo.
echo You can now run Stitch2Stitch with:
echo   launch_windows.bat
echo.

pause










