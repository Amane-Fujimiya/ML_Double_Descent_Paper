# ML Double Descent Paper - Complete Setup Guide

## 📋 Project Overview

This is a **Machine Learning Research Project** investigating the "Double Descent" phenomenon in neural networks using a Non-Equilibrium Statistical Physics (NESP) framework.

**Components:**
- 🧪 **Experiments**: 9 machine learning experiments using PyTorch
- 📄 **Paper**: Academic paper in LaTeX (Cambridge University Press format)
- 💾 **Data**: Research outputs and analysis results
- 🖥️ **GPU**: Optimized for NVIDIA CUDA (you have RTX 3070 Ti - excellent!)

---

## 🚀 Quick Start (5 minutes)

### For Windows CMD Users:
```batch
# Step 1: Python setup
setup.bat

# Step 2: LaTeX setup
setup_miktex.bat
```

### For PowerShell Users:
```powershell
# Step 1: Python setup
.\setup.ps1

# Step 2: LaTeX setup
.\setup_miktex.ps1
```

**Both scripts will:**
1. Create a Python virtual environment
2. Install PyTorch with CUDA support
3. Install all required Python packages
4. Install LaTeX packages
5. Verify everything works

---

## 📦 What Gets Installed

### Python Packages:
| Package | Purpose |
|---------|---------|
| **torch** | Deep learning framework |
| **torchvision, torchaudio** | PyTorch utilities |
| **numpy** | Numerical computing |
| **scipy** | Scientific algorithms |
| **matplotlib** | Data visualization |
| **scikit-learn** | Machine learning utilities |
| **tqdm** | Progress bars |
| **pandas** | Data handling |
| **jupyter** | Interactive notebooks (optional) |

### LaTeX Packages:
- Math: `amsbsy`, `amssymb`, `amsmath`, `amsfonts`, `amsthm`, `braket`
- Graphics: `tikz`, `pgfplots`, `graphicx`, `xcolor`
- Tables: `nicematrix`, `tabularx`, `booktabs`
- And 30+ more packages automatically installed by MikTeX

---

## 🔧 Detailed Setup Instructions

### Option 1: Automated Setup (Recommended)

#### Windows Command Prompt:
```batch
cd C:\Users\Administrator\Documents\GitHub\ML_Double_Descent_Paper\main\paper2

REM Python setup
setup.bat

REM Then MikTeX setup
setup_miktex.bat
```

#### Windows PowerShell:
```powershell
cd C:\Users\Administrator\Documents\GitHub\ML_Double_Descent_Paper\main\paper2

# Python setup
.\setup.ps1

# Then MikTeX setup (run as Administrator)
.\setup_miktex.ps1
```

### Option 2: Manual Setup

#### Python:

**Using conda (recommended for ML):**
```bash
# Create environment
conda create -n ml-descent python=3.10

# Activate
conda activate ml-descent

# Install PyTorch with CUDA 12.1
conda install pytorch::pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other packages
conda install numpy scipy matplotlib scikit-learn tqdm pandas jupyter
```

**Using pip:**
```bash
# Create virtual environment
python -m venv ml-descent-env

# Activate (Windows)
ml-descent-env\Scripts\activate

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other packages
pip install -r requirements.txt
```

#### LaTeX:

**Automatic (recommended):**
```bash
# Just compile - MikTeX will auto-install missing packages
latexmk -pdf paper2a.tex
```

**Manual:**
1. Open MikTeX Console
2. Enable "Install missing packages on-the-fly"
3. Click Settings → Update DB
4. Compile your paper

---

## ▶️ Running Experiments

### Activate Python Environment First:

**cmd.exe:**
```batch
ml-descent-env\Scripts\activate.bat
```

**PowerShell:**
```powershell
.\ml-descent-env\Scripts\Activate.ps1
```

### Run Experiments:

```bash
cd experiments

# Quick test (5-10 minutes)
python run_all.py --quick

# Full suite (30-60 minutes)
python run_all.py

# Specific experiments
python run_all.py --exp 1,4,5

# Individual experiments
python run_exp1_curvature_noise.py --quick
python run_exp2_escape_time.py --quick
python run_exp3_batch_size.py --quick
python run_exp4_relu.py --quick
python run_exp5_erosion.py --quick
```

### Results:
- Plots: `experiments/outputs/*.pdf`
- Data: `experiments/outputs/*.csv`
- Report: `outputs/summary_report.txt`

---

## 📝 Compiling the Paper

### Using latexmk (fastest):
```bash
latexmk -pdf paper2a.tex

# Clean auxiliary files afterward
latexmk -c
```

### Using pdflatex directly:
```bash
pdflatex -interaction=nonstopmode paper2a.tex
bibtex paper2a
pdflatex -interaction=nonstopmode paper2a.tex
pdflatex -interaction=nonstopmode paper2a.tex
```

### Output:
- **Main paper**: `paper2a.pdf`
- **Revised version**: `paper2a_revised.pdf`

---

## ✅ Verification

### Check Python Installation:
```bash
# Activate environment first
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

### Check LaTeX Installation:
```bash
miktex --version
pdflatex --version
bibtex --version
```

### Test GPU Access:
```bash
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU name: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"N/A\"}')"
```

Your RTX 3070 Ti should show:
```
GPU available: True
GPU name: NVIDIA GeForce RTX 3070 Ti
```

---

## 📂 Directory Structure

```
paper2/
├── setup.bat / setup.ps1              ← Python setup scripts
├── setup_miktex.bat / setup_miktex.ps1 ← LaTeX setup scripts
├── SETUP_GUIDE.md                     ← Detailed Python guide
├── MIKTEX_SETUP.md                    ← Detailed LaTeX guide
├── requirements.txt                   ← Python packages list
├── research_notebook.md               ← Research notes
├── paper2a.tex                        ← Main paper (revised)
├── preamble.tex                       ← LaTeX preamble
├── math_commands.tex                  ← Custom math macros
├── references.bib                     ← Bibliography
├── experiments/
│   ├── run_all.py                     ← Master runner
│   ├── run_exp1_curvature_noise.py
│   ├── run_exp2_escape_time.py
│   ├── run_exp3_batch_size.py
│   ├── run_exp4_relu.py
│   ├── run_exp5_erosion.py
│   ├── run_exp6_activation_comparison.py
│   ├── run_exp7_heterogeneity.py
│   ├── run_exp8_sharpness_gradient.py
│   ├── run_exp9_lr_modulation.py
│   ├── models.py
│   ├── utils.py
│   ├── custom_optimizers.py
│   ├── outputs/                       ← Experiment results
│   └── *.py
├── outputs/                           ← Final outputs
├── media/                             ← Figures for paper
└── ml-descent-env/                    ← Virtual environment (created during setup)
```

---

## 🐛 Troubleshooting

### Python Issues

**"No module named 'torch'"**
```bash
# Make sure virtual environment is activated
conda activate ml-descent  # or source ml-descent-env/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**"CUDA not available"**
```bash
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Experiments won't run**
```bash
# Check if you're in the experiments directory
cd experiments

# Check Python path
python -c "import sys; print(sys.path)"

# Make sure all imports work
python -c "from models import LinearTeacherStudent; from utils import train_sgd"
```

### LaTeX Issues

**"cup-journal.cls not found"**
- Make sure you're compiling from the project root directory
- Check that file exists: `ls -la cup-journal.cls`

**"Package not found" error**
```bash
# Enable auto-installation in MikTeX Console or run
initexmf --admin --update-fndb
latexmk -pdf paper2a.tex
```

**Bibliography not appearing**
```bash
# Run full compilation sequence
pdflatex paper2a.tex
bibtex paper2a
pdflatex paper2a.tex
pdflatex paper2a.tex
```

**Very slow compilation**
```bash
# Increase TeX memory
initexmf --admin --set-config-value [Core]main_memory=5000000

# Then recompile
latexmk -pdf paper2a.tex
```

### GPU Issues

**GPU not being used**
```bash
# Check in your Python code or after experiments
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\"}')"
```

**CUDA version mismatch**
```bash
# Check CUDA version
nvidia-smi  # Shows driver version

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Install matching version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 📚 Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed Python setup guide
- **[MIKTEX_SETUP.md](MIKTEX_SETUP.md)** - Detailed LaTeX setup guide
- **[research_notebook.md](research_notebook.md)** - Research notes and findings
- **[requirements.txt](requirements.txt)** - Python packages list
- **paper2a.tex** - Main paper source code

---

## 🎓 Key Experiments

| Exp | Name | Duration | Purpose |
|-----|------|----------|---------|
| 1 | Curvature-Noise Coupling | ~5 min | Verify Σ(W) ∝ H(W) relationship |
| 2 | Escape Time | ~3 min | Test "survival of the flattest" |
| 3 | Batch Size Dependence | ~5 min | Study noise strength effects |
| 4 | ReLU Alignment | ~5 min | Nonlinearity impact on coupling |
| 5 | Equilibrium Erosion | ~5 min | Post-convergence dynamics |
| 6+ | Advanced experiments | ~30 min | Additional phenomena studies |

**Quick mode** reduces model size and epochs (5-10 minutes total)  
**Full mode** runs complete analysis (30-60 minutes on RTX 3070 Ti)

---

## 🖥️ System Specs

Your system specifications:
- **GPU**: NVIDIA RTX 3070 Ti (8 GB VRAM) ✓ Excellent
- **CUDA**: 12.9 ✓ Modern
- **Python**: 3.8+ required
- **RAM**: 8 GB minimum (more is better)
- **Disk**: ~15 GB for full installation

**Your system meets all requirements and is well-optimized!**

---

## ❓ FAQ

**Q: Do I need both conda and pip?**  
A: No. Choose one:
- Conda: Better for ML projects, handles binary dependencies
- pip: Faster, more lightweight

**Q: Can I use different Python versions?**  
A: Python 3.8-3.12 works. 3.10 is recommended.

**Q: Will experiments run on CPU?**  
A: Yes, but much slower. GPU is ~10-100x faster.

**Q: Can I run experiments in parallel?**  
A: Experiments use GPU, so one at a time. But you can modify code.

**Q: How much disk space needed?**  
A: ~15 GB (PyTorch: 4 GB, MikTeX: 2 GB, Project: 1 GB)

**Q: Can I use WSL (Windows Subsystem for Linux)?**  
A: Yes, WSL2 supports CUDA. Similar setup procedure.

---

## 🆘 Getting Help

1. **Check the detailed guides:**
   - Python: [SETUP_GUIDE.md](SETUP_GUIDE.md)
   - LaTeX: [MIKTEX_SETUP.md](MIKTEX_SETUP.md)

2. **Check research notes:**
   - [research_notebook.md](research_notebook.md)

3. **Common issues:** See Troubleshooting section above

4. **External resources:**
   - PyTorch: https://pytorch.org/
   - MikTeX: https://miktex.org/
   - LaTeX: https://www.latex-project.org/

---

## 📝 Notes

- **First Python setup**: ~10 minutes (downloading PyTorch)
- **First LaTeX compilation**: ~15 minutes (downloading packages)
- **Subsequent runs**: Much faster (2-3 minutes)
- **Full experiment suite**: 30-60 minutes on RTX 3070 Ti

---

## 🎉 Ready to Go!

You're all set up! Now you can:

1. ✅ Run experiments: `python run_all.py --quick`
2. ✅ Compile paper: `latexmk -pdf paper2a.tex`
3. ✅ Analyze results: Check `outputs/summary_report.txt`
4. ✅ Review research: Open `research_notebook.md`

**Happy researching!** 🔬

---

**Last Updated**: 2026-05-05  
**Project**: ML Double Descent Paper  
**Author**: Bui Gia Khanh, Luong Van Tam, Dang Tri Trung
