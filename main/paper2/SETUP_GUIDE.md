# Setup Guide - ML Double Descent Paper Project

## Project Overview
- **Type**: Machine Learning Research with LaTeX Documentation
- **Language**: Python (PyTorch) + LaTeX (MikTeX)
- **GPU**: NVIDIA RTX 3070 Ti (CUDA 12.9)
- **Main Components**:
  - Experiments: 9 machine learning experiments on double descent phenomena
  - Paper: Academic paper in LaTeX (CUP journal format)
  - Models: Linear teacher-student and ReLU networks

---

## 1. PYTHON DEPENDENCIES

### Required Packages:
- **torch** (PyTorch) - Deep learning framework
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **scipy** - Scientific computing
- **scikit-learn** (optional but recommended)
- **tqdm** (optional - for progress bars)

### Installation Steps:

#### Option A: Using conda (Recommended for ML projects)
```bash
# Create virtual environment
conda create -n ml-descent python=3.10

# Activate environment
conda activate ml-descent

# Install PyTorch with CUDA 12.1 support
conda install pytorch::pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install scientific packages
conda install numpy scipy matplotlib scikit-learn tqdm pandas -c conda-forge

# Install Jupyter (optional, for notebooks)
conda install jupyter jupyterlab -c conda-forge
```

#### Option B: Using pip
```bash
# Create virtual environment
python -m venv ml-descent-env

# Activate environment (Windows)
ml-descent-env\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install scientific packages
pip install numpy scipy matplotlib scikit-learn tqdm pandas

# Install Jupyter (optional)
pip install jupyter jupyterlab
```

---

## 2. MIKTEX PACKAGES

### Required LaTeX Packages:
The project uses the following packages (already listed in `preamble.tex`):

**Math & Symbols:**
- amsbsy, amssymb, amsmath, amsfonts, amsthm
- braket, stmaryrd, nicefrac, microtype
- bbm, bm

**Graphics & Drawing:**
- tikz (with libraries: matrix, patterns, shadings, shapes.geometric, calc, positioning, fit)
- tikz-cd, tikz-3dplot
- circuitikz
- graphicx, pgfplots
- xcolor (with dvipsnames,svgnames,x11names)

**Tables & Lists:**
- nicematrix, tabularx, array
- booktabs, multirow
- enumitem

**Formatting:**
- caption, subcaption
- float, fancyhdr, parskip
- algorithm2e, algorithmicx
- mathtools, parnotes
- thmtools, url
- csquotes, inputenc (utf-8)

**Journal-Specific:**
- cup-journal.cls (Cambridge University Press)
- apj.bst (bibliography style)

### MikTeX Installation:

#### Option A: Automatic Package Management (Recommended)
MikTeX has "On-the-fly installation" enabled by default. When you compile a `.tex` file, missing packages are automatically downloaded and installed.

```bash
# Just compile the paper - MikTeX will auto-install missing packages
latexmk -pdf paper2a.tex
```

#### Option B: Manual Installation via MikTeX Console
1. Open **MikTeX Console** (installed with MikTeX)
2. Switch to "Always install missing packages on-the-fly" mode
3. Click Update (in Maintenance tab)

#### Option C: Using Command Line (Windows PowerShell)
```powershell
# List all installed packages
initexmf --admin --print-only

# Force update of package database
initexmf --admin --update-fndb

# Install specific package (if needed)
tlmgr install <package-name>  # Note: tlmgr works better on TeXLive, not MikTeX
```

---

## 3. PROJECT STRUCTURE

```
paper2/
├── experiments/
│   ├── models.py                 # Model definitions
│   ├── utils.py                  # Training & analysis utilities
│   ├── custom_optimizers.py      # SGD implementations
│   ├── run_*.py                  # Experiment scripts
│   ├── outputs/                  # Experiment results
│   └── run_all.py               # Master runner
├── paper2a.tex                   # Main paper (revised version)
├── preamble.tex                  # LaTeX preamble
├── math_commands.tex             # Custom math macros
├── references.bib                # Bibliography
├── SETUP_GUIDE.md               # This file
└── outputs/                      # Compiled papers & reports

```

---

## 4. QUICK START

### Step 1: Set up Python environment
```bash
# Using conda (recommended)
conda create -n ml-descent python=3.10
conda activate ml-descent
conda install pytorch::pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install numpy scipy matplotlib scikit-learn tqdm
```

### Step 2: Test Python installation
```bash
# Run a quick experiment
cd experiments
python run_exp1_curvature_noise.py --quick

# Or run all experiments
python run_all.py --quick --output ./test_outputs
```

### Step 3: Compile LaTeX paper
```bash
# In project root directory
latexmk -pdf paper2a.tex

# Or using pdflatex directly
pdflatex -interaction=nonstopmode paper2a.tex
bibtex paper2a
pdflatex -interaction=nonstopmode paper2a.tex
pdflatex -interaction=nonstopmode paper2a.tex
```

### Step 4: View outputs
- **Paper**: `paper2a.pdf`
- **Experiment results**: `outputs/summary_report.txt`
- **Plots**: `experiments/outputs/*.pdf`

---

## 5. SYSTEM REQUIREMENTS

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.10+ |
| CUDA | 11.8 | 12.1+ |
| RAM | 8 GB | 16 GB+ |
| GPU VRAM | 2 GB | 4 GB+ (your RTX 3070 Ti has 8 GB ✓) |
| Disk Space | 5 GB | 15 GB+ |

Your system (RTX 3070 Ti with CUDA 12.9) **exceeds all requirements** ✓

---

## 6. TROUBLESHOOTING

### Python Issues

**Issue: CUDA not detected**
```bash
# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Issue: Module not found errors**
```bash
# Make sure you're in the correct Python environment
conda activate ml-descent  # or source activation script

# Reinstall packages
pip install --upgrade --force-reinstall torch numpy scipy matplotlib
```

### LaTeX Issues

**Issue: `cup-journal.cls` not found**
- This file is in the project root. Make sure you compile from the project directory.

**Issue: Missing TikZ libraries**
```powershell
# MikTeX should auto-install, but if not:
# Open MikTeX Console > Packages > Search for "pgfplots"
# Right-click > Install
```

**Issue: Bibliography not appearing**
```bash
# Make sure to run bibtex
pdflatex paper2a.tex
bibtex paper2a
pdflatex paper2a.tex
pdflatex paper2a.tex  # Run twice for references to resolve
```

---

## 7. RUNNING EXPERIMENTS

### Individual Experiments
```bash
cd experiments

# Exp 1: Curvature-Noise Coupling
python run_exp1_curvature_noise.py --quick

# Exp 2: Escape Time
python run_exp2_escape_time.py --quick

# Exp 3: Batch Size Dependence
python run_exp3_batch_size.py --quick

# Exp 4: ReLU Alignment
python run_exp4_relu.py --quick

# Exp 5: Equilibrium Erosion
python run_exp5_erosion.py --quick
```

### Run All Experiments
```bash
# Quick mode (for testing)
python run_all.py --quick

# Full mode (requires ~30-60 min on RTX 3070 Ti)
python run_all.py

# Run specific experiments
python run_all.py --exp 1,4,5
```

### Command-line Options
```bash
--quick              # Use smaller models, fewer epochs
--exp 1,2,3         # Run only experiments 1, 2, 3
--output ./results  # Custom output directory
```

---

## 8. NEXT STEPS

1. ✓ Install Python packages (Step 1)
2. ✓ Test experiments (Step 2)
3. ✓ Compile paper (Step 3)
4. ✓ Review outputs (Step 4)
5. Modify experiments or paper as needed

---

## Useful Commands

### Update all packages
```bash
conda update --all  # or pip install --upgrade --all
```

### Check installed versions
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

### List all Python dependencies
```bash
pip list
# or
conda list
```

---

**Last Updated**: 2026-05-05  
**Created for**: ML Double Descent Research Paper
