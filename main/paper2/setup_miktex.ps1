#!/usr/bin/env powershell
# ============================================================================
# MikTeX Setup Script for ML Double Descent Paper (PowerShell)
#
# This script updates MikTeX and ensures all required packages are installed
#
# Usage: 
#   .\setup_miktex.ps1                # Full setup and compile
#   .\setup_miktex.ps1 -SkipCompile   # Setup without compiling
#
# Requirements:
#   - MikTeX must be already installed
#   - Run as Administrator for package installation
#
# Note: If you get execution policy error, run:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# ============================================================================

param(
    [switch]$SkipCompile = $false,
    [switch]$Verbose = $false
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

function Write-Package {
    param([string]$Package)
    Write-Host "    • $Package"
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Main setup
Write-Header "MikTeX Setup for ML Double Descent Paper"

# Check administrator privileges
if (-not (Test-Administrator)) {
    Write-Warning "Not running as Administrator. Package installation may fail."
    Write-Host "To run as Administrator:" -ForegroundColor $Yellow
    Write-Host "  1. Right-click PowerShell"
    Write-Host "  2. Select 'Run as Administrator'"
    Write-Host "  3. Run: .\setup_miktex.ps1"
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/n)"
    if ($response -ne "y") { exit }
}

Write-Host ""

# Step 1: Check MikTeX installation
Write-Step "Checking MikTeX installation" 1 3

try {
    $miktexVersion = miktex --version 2>$null
    Write-Success "MikTeX is installed"
    Write-Host "  Version: $miktexVersion" -ForegroundColor $Cyan
} catch {
    Write-Error "MikTeX is not installed or not in PATH"
    Write-Host ""
    Write-Host "Please install MikTeX from: https://miktex.org/" -ForegroundColor $Yellow
    exit 1
}

Write-Host ""

# Step 2: Update package database and install packages
Write-Step "Installing/Updating essential packages" 2 3

try {
    Write-Host "  Updating package database..." -ForegroundColor $Yellow
    initexmf --admin --update-fndb 2>$null
    Write-Success "Package database updated"
} catch {
    Write-Warning "Package database update may have had issues"
}

Write-Host ""
Write-Host "  Installing packages:" -ForegroundColor $Cyan

# List of packages to install
$packages = @(
    "amsmath", "amssymb", "amsfonts", "amsbsy", "amsthm",
    "pgfplots", "tikz", "tikz-cd", "circuitikz",
    "graphicx", "xcolor", "braket", "stmaryrd", "nicefrac", "microtype", "bbm", "bm",
    "caption", "subcaption", "float", "fancyhdr", "parskip", "enumitem",
    "mathtools", "parnotes", "nicematrix", "tabularx", "array", "booktabs", "multirow",
    "thmtools", "url", "csquotes", "inputenc",
    "biblatex", "bibtex",
    "algorithm2e", "algorithmicx",
    "geometry", "fancybox", "listings"
)

$successCount = 0
$failCount = 0

foreach ($package in $packages) {
    Write-Package $package
    try {
        initexmf --admin --package=$package 2>$null | Out-Null
        $successCount++
    } catch {
        Write-Warning "Failed to install: $package"
        $failCount++
    }
}

Write-Host ""
Write-Success "Installed/Updated $successCount packages"
if ($failCount -gt 0) {
    Write-Warning "$failCount packages had issues (may be already installed)"
}

Write-Host ""

# Step 3: Compile paper
if ($SkipCompile) {
    Write-Host "Skipping paper compilation as requested" -ForegroundColor $Yellow
} else {
    Write-Step "Compiling paper" 3 3
    Write-Host "  This may take a few minutes on first compilation..." -ForegroundColor $Yellow
    Write-Host ""

    # Check if latexmk exists
    $latexmkPath = (Get-Command latexmk -ErrorAction SilentlyContinue).Source

    if ($latexmkPath) {
        Write-Host "  Using latexmk (faster)..." -ForegroundColor $Cyan
        & latexmk -pdf paper2a.tex 2>&1 | Select-String -Pattern "Output" -Context 0,5
    } else {
        Write-Warning "latexmk not found, using pdflatex (slower)"
        Write-Host "  Running pdflatex..." -ForegroundColor $Cyan

        $steps = @(
            "pdflatex -interaction=nonstopmode paper2a.tex",
            "bibtex paper2a",
            "pdflatex -interaction=nonstopmode paper2a.tex",
            "pdflatex -interaction=nonstopmode paper2a.tex"
        )

        foreach ($step in $steps) {
            Invoke-Expression $step 2>&1 | Select-String -Pattern "Output file removed" -Context 0,2
        }
    }
}

Write-Host ""

# Final summary
Write-Header "MikTeX Setup Complete!"

if (Test-Path "paper2a.pdf") {
    Write-Success "Paper compiled successfully: paper2a.pdf"
} else {
    Write-Warning "Paper may not have compiled successfully"
}

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor $Cyan
Write-Host "  1. Verify compilation was successful" -ForegroundColor $Yellow
Write-Host "     Open: paper2a.pdf"
Write-Host ""
Write-Host "  2. To recompile the paper in the future:" -ForegroundColor $Yellow
Write-Host "     latexmk -pdf paper2a.tex"
Write-Host ""
Write-Host "  3. For Python experiments:" -ForegroundColor $Yellow
Write-Host "     cd experiments"
Write-Host "     python run_all.py --quick"
Write-Host ""

Write-Host "Setup complete! Happy researching!" -ForegroundColor $Green
Write-Host ""
