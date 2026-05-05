#!/usr/bin/env powershell
# ============================================================================
# SETUP SCRIPT for ML Double Descent Paper Project (Windows PowerShell)
# 
# This script:
#   1. Creates a Python virtual environment
#   2. Installs PyTorch with CUDA 12.1 support
#   3. Installs all required packages
#   4. Verifies the installation
#   5. (Optional) Runs a quick test
#
# Usage: 
#   .\setup.ps1                  # Full setup
#   .\setup.ps1 -Quick           # Setup + quick test
#   .\setup.ps1 -RunTests        # Run full test suite
#
# Note: If you get execution policy error, run:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# ============================================================================

param(
    [switch]$Quick = $false,
    [switch]$RunTests = $false,
    [string]$PythonVersion = "3.10"
)

# Set error action preference
$ErrorActionPreference = "Continue"

# Colors for output
$Green = "Green"
$Red = "Red"
$Yellow = "Yellow"
$Cyan = "Cyan"

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor $Cyan
    Write-Host $Message -ForegroundColor $Cyan
    Write-Host "============================================================================" -ForegroundColor $Cyan
    Write-Host ""
}

function Write-Step {
    param([string]$Message, [int]$StepNum, [int]$TotalSteps)
    Write-Host "[$StepNum/$TotalSteps] $Message" -ForegroundColor $Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "  ✓ $Message" -ForegroundColor $Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "  ✗ ERROR: $Message" -ForegroundColor $Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "  ⚠ $Message" -ForegroundColor $Yellow
}

# Main setup
Write-Header "ML Double Descent Paper - Environment Setup (PowerShell)"

# Step 1: Check Python installation
Write-Step "Checking Python installation" 1 5
try {
    $pythonVersion = python --version 2>&1
    Write-Success "Python is installed: $pythonVersion"
} catch {
    Write-Error "Python is not installed or not in PATH"
    Write-Host ""
    Write-Host "Please install Python 3.8+ from: https://www.python.org/" -ForegroundColor $Yellow
    exit 1
}

Write-Host ""

# Step 2: Create virtual environment
Write-Step "Creating Python virtual environment" 2 5
$venvPath = "ml-descent-env"

if (Test-Path $venvPath) {
    Write-Warning "Virtual environment already exists at: $venvPath"
} else {
    try {
        python -m venv $venvPath
        Write-Success "Created virtual environment: $venvPath"
    } catch {
        Write-Error "Failed to create virtual environment"
        exit 1
    }
}

Write-Host ""

# Step 3: Activate virtual environment
Write-Step "Activating virtual environment" 3 5
$activateScript = "$venvPath\Scripts\Activate.ps1"

if (-not (Test-Path $activateScript)) {
    Write-Error "Activation script not found at: $activateScript"
    exit 1
}

try {
    & $activateScript
    Write-Success "Activated: $venvPath"
} catch {
    Write-Error "Failed to activate virtual environment"
    Write-Host "You may need to enable PowerShell execution policy:" -ForegroundColor $Yellow
    Write-Host "  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor $Yellow
    exit 1
}

Write-Host ""

# Step 4: Install packages
Write-Step "Installing packages" 4 5

try {
    Write-Host "  Installing pip, setuptools, wheel..."
    python -m pip install --upgrade pip setuptools wheel --quiet 2>$null
    Write-Success "pip, setuptools, wheel upgraded"
} catch {
    Write-Warning "pip upgrade had issues, continuing..."
}

try {
    Write-Host "  Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
    Write-Success "PyTorch installed"
} catch {
    Write-Error "Failed to install PyTorch"
    exit 1
}

try {
    Write-Host "  Installing scientific packages (numpy, scipy, matplotlib, scikit-learn, tqdm, pandas)..."
    pip install numpy scipy matplotlib scikit-learn tqdm pandas --quiet
    Write-Success "Scientific packages installed"
} catch {
    Write-Error "Failed to install scientific packages"
    exit 1
}

try {
    Write-Host "  Installing Jupyter (optional)..."
    pip install jupyter jupyterlab --quiet 2>$null
    Write-Success "Jupyter installed"
} catch {
    Write-Warning "Jupyter installation skipped"
}

Write-Host ""

# Step 5: Verify installation
Write-Step "Verifying installation" 5 5

$torchVersion = python -c "import torch; print(torch.__version__)" 2>$null
$cudaAvailable = python -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')" 2>$null
$cudaVersion = python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>$null
$numpyVersion = python -c "import numpy; print(numpy.__version__)" 2>$null
$scipyVersion = python -c "import scipy; print(scipy.__version__)" 2>$null
$matplotlibVersion = python -c "import matplotlib; print(matplotlib.__version__)" 2>$null
$sklearnVersion = python -c "import sklearn; print(sklearn.__version__)" 2>$null

Write-Host "  Package Versions:" -ForegroundColor $Cyan
Write-Host "    PyTorch:       $torchVersion"
Write-Host "    CUDA Available: $cudaAvailable"
Write-Host "    CUDA Version:  $cudaVersion"
Write-Host "    NumPy:         $numpyVersion"
Write-Host "    SciPy:         $scipyVersion"
Write-Host "    Matplotlib:    $matplotlibVersion"
Write-Host "    Scikit-learn:  $sklearnVersion"
Write-Host ""

# Summary and next steps
Write-Header "Setup Complete!"

Write-Host "Environment Details:" -ForegroundColor $Cyan
Write-Host "  Virtual Environment: $(Resolve-Path $venvPath)"
Write-Host "  Python Executable:   $(Resolve-Path $venvPath\Scripts\python.exe)"
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor $Cyan
Write-Host ""
Write-Host "1. To activate the environment in the future:" -ForegroundColor $Yellow
Write-Host "   .\ml-descent-env\Scripts\Activate.ps1"
Write-Host ""
Write-Host "2. To run experiments:" -ForegroundColor $Yellow
Write-Host "   cd experiments"
Write-Host "   python run_all.py --quick     # Quick test (5-10 min)"
Write-Host "   python run_all.py              # Full suite (30-60 min)"
Write-Host ""
Write-Host "3. To run individual experiments:" -ForegroundColor $Yellow
Write-Host "   python run_exp1_curvature_noise.py --quick"
Write-Host "   python run_exp2_escape_time.py --quick"
Write-Host "   python run_exp3_batch_size.py --quick"
Write-Host ""
Write-Host "4. To compile the LaTeX paper:" -ForegroundColor $Yellow
Write-Host "   latexmk -pdf paper2a.tex"
Write-Host ""
Write-Host "5. To view the research notebook:" -ForegroundColor $Yellow
Write-Host "   Start-Process research_notebook.md  # Open in default app"
Write-Host ""

# Optional: Run quick test
if ($Quick) {
    Write-Host ""
    Write-Host "Running quick test..." -ForegroundColor $Cyan
    Write-Host ""
    Push-Location experiments
    python run_all.py --quick --output ..\outputs\quick_test
    Pop-Location
}

Write-Host ""
Write-Host "Setup script finished. Happy researching!" -ForegroundColor $Green
Write-Host ""
