@echo off
REM ============================================================================
REM SETUP SCRIPT for ML Double Descent Paper Project (Windows)
REM 
REM This script:
REM   1. Creates a Python virtual environment
REM   2. Installs PyTorch with CUDA 12.1 support
REM   3. Installs all required packages
REM   4. Verifies the installation
REM   5. (Optional) Runs a quick test
REM
REM Usage: 
REM   setup.bat                  # Full setup
REM   setup.bat quick            # Setup + quick test
REM
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo ML Double Descent Paper - Environment Setup (Windows)
echo ============================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/5] Python version:
python --version
echo.

REM Create virtual environment
echo [2/5] Creating Python virtual environment...
if exist ml-descent-env (
    echo   Virtual environment already exists, skipping creation
) else (
    python -m venv ml-descent-env
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo   Created: ml-descent-env
)
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call ml-descent-env\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo   Activated: ml-descent-env
echo   Python: !VIRTUAL_ENV!\Scripts\python.exe
echo.

REM Upgrade pip
echo [4/5] Upgrading pip and installing packages...
echo   Upgrading pip...
python -m pip install --upgrade pip setuptools wheel --quiet
if errorlevel 1 (
    echo   Warning: pip upgrade had issues, continuing...
)

echo   Installing PyTorch with CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo   Installing scientific packages...
pip install numpy scipy matplotlib scikit-learn tqdm pandas --quiet
if errorlevel 1 (
    echo ERROR: Failed to install scientific packages
    pause
    exit /b 1
)

echo   Installing Jupyter (optional)...
pip install jupyter jupyterlab --quiet

echo   Installation complete!
echo.

REM Verify installation
echo [5/5] Verifying installation...
echo.

python -c "import torch; print('  PyTorch version:', torch.__version__)"
python -c "import torch; cuda_available = torch.cuda.is_available(); print('  CUDA available:', cuda_available); print('  CUDA version:', torch.version.cuda if cuda_available else 'N/A')"
python -c "import numpy; print('  NumPy version:', numpy.__version__)"
python -c "import scipy; print('  SciPy version:', scipy.__version__)"
python -c "import matplotlib; print('  Matplotlib version:', matplotlib.__version__)"
python -c "import sklearn; print('  Scikit-learn version:', sklearn.__version__)"

echo.
echo ============================================================================
echo Setup Complete!
echo ============================================================================
echo.
echo To activate the environment in the future, run:
echo   ml-descent-env\Scripts\activate.bat
echo.
echo To run experiments, execute:
echo   cd experiments
echo   python run_all.py --quick
echo.
echo Or run individual experiments:
echo   python run_exp1_curvature_noise.py --quick
echo   python run_exp2_escape_time.py --quick
echo   etc.
echo.
echo To compile the LaTeX paper, run:
echo   latexmk -pdf paper2a.tex
echo.

REM Optional: Run quick test
if "%1"=="quick" (
    echo Running quick test...
    echo.
    cd experiments
    python run_all.py --quick --output ..\outputs\quick_test
    cd ..
)

echo.
pause
