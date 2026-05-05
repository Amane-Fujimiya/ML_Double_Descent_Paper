@echo off
REM ============================================================================
REM MikTeX Setup Script for ML Double Descent Paper
REM
REM This script updates MikTeX and ensures all required packages are installed
REM
REM Usage: 
REM   setup_miktex.bat                # Full setup
REM
REM Requirements:
REM   - MikTeX must be already installed
REM   - Administrator privileges recommended
REM
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo MikTeX Setup for ML Double Descent Paper
echo ============================================================================
echo.

REM Check if MikTeX is installed
miktex --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: MikTeX is not installed or not in PATH
    echo Please install MikTeX from https://miktex.org/
    echo.
    pause
    exit /b 1
)

echo MikTeX Version:
miktex --version
echo.

REM Update package database
echo [1/3] Updating package database...
call initexmf --admin --update-fndb
echo   Done.
echo.

REM Install essential packages
echo [2/3] Installing/Updating essential packages...
echo.

setlocal DisableDelayedExpansion

set "packages=amsmath amssymb amsfonts amsbsy amsthm pgfplots tikz tikz-cd circuitikz graphicx xcolor braket stmaryrd nicefrac microtype bbm bm caption subcaption float fancyhdr parskip enumitem mathtools parnotes nicematrix tabularx array booktabs multirow thmtools url csquotes inputenc biblatex bibtex algorithm2e algorithmicx"

for %%P in (%packages%) do (
    echo   Installing %%P...
    call initexmf --admin --package=%%P >nul 2>&1
)

echo   All packages installed.
echo.

REM Compile paper with latexmk
echo [3/3] Compiling paper (first-time compilation may take a few minutes)...
echo.

if exist latexmk.exe (
    latexmk -pdf paper2a.tex
) else (
    echo   Using pdflatex (slower - consider installing latexmk)
    pdflatex -interaction=nonstopmode paper2a.tex
    bibtex paper2a
    pdflatex -interaction=nonstopmode paper2a.tex
    pdflatex -interaction=nonstopmode paper2a.tex
)

echo.
echo ============================================================================
echo MikTeX Setup Complete!
echo ============================================================================
echo.

if exist paper2a.pdf (
    echo SUCCESS: Paper compiled as paper2a.pdf
) else (
    echo WARNING: Paper compilation may have had issues
)

echo.
echo To recompile the paper in the future, run:
echo   latexmk -pdf paper2a.tex
echo.
echo Or use pdflatex manually:
echo   pdflatex paper2a.tex
echo   bibtex paper2a
echo   pdflatex paper2a.tex
echo   pdflatex paper2a.tex
echo.

pause
