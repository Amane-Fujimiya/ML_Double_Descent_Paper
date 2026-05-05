# MikTeX Setup Guide for ML Double Descent Paper

## Quick Summary

Your LaTeX paper requires several packages. **MikTeX has automatic package installation by default** - simply compile and missing packages will be installed automatically.

---

## Automatic Installation (Recommended)

MikTeX automatically installs missing packages when you compile. Just run:

```bash
# From project root directory
latexmk -pdf paper2a.tex

# Or using pdflatex directly (slower)
pdflatex -interaction=nonstopmode paper2a.tex
bibtex paper2a
pdflatex -interaction=nonstopmode paper2a.tex
pdflatex -interaction=nonstopmode paper2a.tex
```

**First compilation may take 5-15 minutes** while MikTeX downloads and installs packages.

---

## Manual Package Installation (If Needed)

If automatic installation fails or you prefer manual control:

### Using MikTeX Console (GUI)

1. **Open MikTeX Console**
   - Search for "MikTeX Console" in Windows Start menu
   - Or run: `miktexconsole.exe`

2. **Switch to Administrator Mode**
   - Click "⚙️ Settings" (gear icon)
   - Under "Preferred installer" select "Always ask"

3. **Enable Auto-Installation**
   - Settings → General → Check "Install missing packages on-the-fly"

4. **Update Package Repository**
   - Tasks → Update DB
   - Tasks → Refresh FNDB

5. **Manual Install (if needed)**
   - Packages → Search for package name
   - Right-click → Install

### Using Command Line (PowerShell)

```powershell
# Check if MikTeX is installed
miktex --version

# Force package database refresh
initexmf --admin --update-fndb

# Install specific package
initexmf --admin --package=<package-name>

# Example: Install pgfplots package
initexmf --admin --package=pgfplots
```

### Common Packages to Install Manually (if auto-install fails)

```powershell
# Core math packages
initexmf --admin --package=amsmath
initexmf --admin --package=amssymb
initexmf --admin --package=amsbsy

# Graphics packages
initexmf --admin --package=tikz
initexmf --admin --package=pgfplots
initexmf --admin --package=tikz-cd

# Bibliography
initexmf --admin --package=biblatex
initexmf --admin --package=bibtex

# Tables & formatting
initexmf --admin --package=nicematrix
initexmf --admin --package=tabularx
initexmk --admin --package=booktabs

# Algorithms
initexmf --admin --package=algorithm2e
initexmf --admin --package=algorithmicx
```

---

## Full Package List (Reference)

The following packages are required and typically auto-installed:

| Category | Packages |
|----------|----------|
| **Math & Symbols** | amsbsy, amssymb, amsmath, amsfonts, amsthm, braket, stmaryrd, nicefrac, microtype, bbm, bm |
| **Graphics & Drawing** | tikz, pgfplots, tikz-cd, tikz-3dplot, circuitikz, graphicx, xcolor, pstricks |
| **Tables & Lists** | nicematrix, tabularx, array, booktabs, multirow, enumitem, longtable |
| **Formatting** | caption, subcaption, float, fancyhdr, parskip, parnotes, mathtools, thmtools |
| **Algorithms** | algorithm2e, algorithmicx |
| **Bibliography** | biblatex, bibtex |
| **Text & Language** | url, csquotes, inputenc, babel, polyglossia |
| **Journal Specific** | cup-journal (included locally) |

---

## Troubleshooting

### Issue 1: "Package not found" error during compilation

**Solution**: Enable auto-installation and update package database

```bash
# In PowerShell/CMD as Administrator
initexmf --admin --package=missing-package-name
initexmf --admin --update-fndb
latexmk -pdf paper2a.tex
```

### Issue 2: "cup-journal.cls not found"

**Solution**: This file is included in the project. Compile from the project root directory:

```bash
# Make sure you're in the project root (paper2/) directory
cd C:\Users\Administrator\Documents\GitHub\ML_Double_Descent_Paper\main\paper2

# Then compile
latexmk -pdf paper2a.tex
```

### Issue 3: TikZ library errors

**Solution**: MikTeX often has TikZ issues. Try updating:

```powershell
# As Administrator
initexmf --admin --update-fndb
initexmf --admin --package=pgfplots
initexmf --admin --package=tikz
```

Then recompile with explicit interaction:

```bash
pdflatex -interaction=nonstopmode paper2a.tex
```

### Issue 4: Bibliography not appearing

**Solution**: Run full compilation sequence:

```bash
pdflatex -interaction=nonstopmode paper2a.tex
bibtex paper2a
pdflatex -interaction=nonstopmode paper2a.tex
pdflatex -interaction=nonstopmode paper2a.tex
```

Or use latexmk:

```bash
latexmk -pdf -bibtex paper2a.tex
```

### Issue 5: Memory/performance issues

If compilation is very slow:

```bash
# Increase TeX memory
initexmf --admin --set-config-value [Core]main_memory=5000000
initexmf --admin --set-config-value [Core]hash_extra=100000

# Then recompile
latexmk -pdf paper2a.tex
```

---

## Installation Scripts

### Batch File (cmd.exe)

Create `setup_miktex.bat` in project root:

```batch
@echo off
echo Installing/Updating MikTeX packages...
echo.

REM Update package database
echo Updating package database...
initexmf --admin --update-fndb

REM Install essential packages
echo Installing core packages...
initexmf --admin --package=amsmath
initexmf --admin --package=pgfplots
initexmf --admin --package=tikz

REM Compile paper
echo.
echo Compiling paper...
latexmk -pdf paper2a.tex

echo.
echo Done! Paper compiled as paper2a.pdf
pause
```

Run with: `setup_miktex.bat`

### PowerShell Script

Create `setup_miktex.ps1`:

```powershell
# Check if running as administrator
if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Please run as Administrator!" -ForegroundColor Red
    exit 1
}

Write-Host "Updating MikTeX..." -ForegroundColor Cyan

# Update database
Write-Host "Updating package database..."
initexmf --admin --update-fndb

# Install packages
$packages = @(
    "amsmath", "amssymb", "pgfplots", "tikz",
    "nicematrix", "biblatex", "algorithm2e", "graphicx"
)

foreach ($pkg in $packages) {
    Write-Host "Installing $pkg..." -ForegroundColor Yellow
    initexmf --admin --package=$pkg
}

Write-Host ""
Write-Host "Compiling paper..." -ForegroundColor Cyan
latexmk -pdf paper2a.tex

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
```

Run with: `.\setup_miktex.ps1` (as Administrator)

---

## Useful MikTeX Commands

```powershell
# Check MikTeX version
miktex --version

# List all installed packages
mpm --list-installed

# Search for specific package
mpm --search "package-name"

# Update all packages
mpm --update

# Package sizes
du -sh "C:\Program Files\MiKTeX*/\texmfs\install"
```

---

## FAQ

**Q: Do I need to install each package manually?**
A: No! MikTeX auto-installs when compiling. Just run `latexmk -pdf paper2a.tex` and let it work.

**Q: How much disk space does MikTeX need?**
A: ~2-4 GB for full install with all packages used by this project.

**Q: Can I use TeXLive instead of MikTeX?**
A: Yes! TeXLive has similar auto-installation features. Setup is similar but commands differ.

**Q: Why is first compilation so slow?**
A: MikTeX downloads packages on-the-fly. Subsequent compilations are much faster.

**Q: How do I know which packages are installed?**
A: Run `mpm --list-installed | findstr tikz` (PowerShell) to search installed packages.

---

## Support

For MikTeX-specific issues:
- Official Site: https://miktex.org/
- Documentation: https://docs.miktex.org/
- Package Repository: https://ctan.org/

For paper-specific issues:
- LaTeX Classes: `cup-journal.cls` (included)
- Bibliography: `references.bib`
- Macros: `math_commands.tex`, `preamble.tex`

---

**Last Updated**: 2026-05-05
