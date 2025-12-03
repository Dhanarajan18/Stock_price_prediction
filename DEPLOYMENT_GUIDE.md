# 📦 Deployment Guide - Stock Price Prediction System

## ✅ Build Status

**Executable successfully created!**  
Location: `dist\StockPredictor.exe`  
Size: ~500MB (includes all dependencies)

---

## 🚀 Quick Start for End Users

### Running the Executable

No Python installation required! Just double-click or use command line:

```powershell
# Interactive Mode (Recommended for beginners)
.\StockPredictor.exe -i

# Command Line Mode - Train and Predict
.\StockPredictor.exe RELIANCE --train --history 365 --days 5

# Quick Prediction (using pre-trained models)
.\StockPredictor.exe TCS --days 3

# LSTM Only Prediction
.\StockPredictor.exe INFY --train --model lstm --days 7
```

### First Time Setup

1. **Extract Files** (if distributed as ZIP):
   ```powershell
   Expand-Archive -Path StockPredictor.zip -DestinationPath C:\StockPredictor
   cd C:\StockPredictor
   ```

2. **Run Interactive Mode**:
   ```powershell
   .\dist\StockPredictor.exe -i
   ```

3. **Follow On-Screen Instructions**:
   - Enter stock symbol (e.g., RELIANCE, TCS, INFY)
   - Choose to train models (recommended first time)
   - View predictions and save results

---

## 🔧 For Developers

### Building from Source

#### Prerequisites

- **Python**: 3.8 or higher
- **pip**: Latest version
- **Git**: For cloning repository
- **Disk Space**: ~2GB for dependencies

#### Step 1: Clone Repository

```powershell
git clone <your-repo-url>
cd Stock_price_prediction
```

#### Step 2: Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt

# Install PyInstaller if not included
pip install pyinstaller==5.13.0
```

#### Step 3: Build Executable

```powershell
# Clean previous builds
Remove-Item -Path ".\build", ".\dist" -Recurse -Force -ErrorAction SilentlyContinue

# Build using spec file
pyinstaller stock_predictor.spec

# Or use one-line command
pyinstaller --name=StockPredictor --onefile --console --hidden-import=scipy._lib.array_api_compat.numpy.fft src\cli_interface.py
```

#### Build Output

```
dist\
  └── StockPredictor.exe    # Standalone executable
build\
  └── stock_predictor\      # Build artifacts (can be deleted)
```

### Build Configuration

The build is configured in `stock_predictor.spec`:

```python
# Key settings:
- Entry point: src\cli_interface.py
- Mode: --onefile (single executable)
- Console: True (shows output window)
- Hidden imports: All required ML libraries
- Icon: Default console icon
```

---

## 📁 Distribution Options

### Option 1: Single EXE (Current)

**Pros**:
- ✅ Easy to distribute (one file)
- ✅ No installation needed
- ✅ Self-contained

**Cons**:
- ❌ Large file size (~500MB)
- ❌ Slower startup time

```powershell
# Just share the exe file
dist\StockPredictor.exe
```

### Option 2: Directory Distribution

**Pros**:
- ✅ Faster startup
- ✅ Smaller individual files

**Cons**:
- ❌ Multiple files to distribute

```powershell
# Modify spec file: change onefile=False
pyinstaller stock_predictor.spec

# Creates:
dist\
  └── StockPredictor\
       ├── StockPredictor.exe
       ├── *.dll files
       └── dependencies
```

### Option 3: Installer Package

Create a professional installer using **Inno Setup** or **NSIS**:

```powershell
# Using Inno Setup (download from jrsoftware.org)
# Create installer script (example):

[Setup]
AppName=Stock Price Predictor
AppVersion=1.0
DefaultDirName={pf}\StockPredictor
OutputDir=installers

[Files]
Source: "dist\StockPredictor.exe"; DestDir: "{app}"

[Icons]
Name: "{commondesktop}\Stock Predictor"; Filename: "{app}\StockPredictor.exe"
```

---

## 📤 Deployment Scenarios

### Scenario 1: Local Desktop Application

```powershell
# 1. Copy exe to program files
Copy-Item "dist\StockPredictor.exe" -Destination "C:\Program Files\StockPredictor\"

# 2. Create desktop shortcut
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$Home\Desktop\Stock Predictor.lnk")
$Shortcut.TargetPath = "C:\Program Files\StockPredictor\StockPredictor.exe"
$Shortcut.Arguments = "-i"
$Shortcut.Save()
```

### Scenario 2: Network Share

```powershell
# 1. Copy to network location
Copy-Item "dist\StockPredictor.exe" -Destination "\\company-server\shared\tools\"

# 2. Users run from network
\\company-server\shared\tools\StockPredictor.exe -i
```

### Scenario 3: USB Drive

```powershell
# 1. Create portable folder structure
E:\
  └── StockPredictor\
       ├── StockPredictor.exe
       ├── data\              # For stored predictions
       ├── models\            # For saved models
       └── README.txt         # Usage instructions
```

---

## 🔒 Security & Antivirus

### Windows SmartScreen Warning

First-time users may see "Windows protected your PC" warning.

**Solution**:
1. Click "More info"
2. Click "Run anyway"

**To prevent this** (for distribution):
1. **Code Sign the Executable**:
   - Purchase code signing certificate
   - Sign with `signtool.exe`:
   ```powershell
   signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com dist\StockPredictor.exe
   ```

2. **Build reputation**: More downloads = fewer warnings

### Antivirus False Positives

ML libraries can trigger false positives.

**Solutions**:
1. **VirusTotal**: Upload and share clean scan results
2. **Whitelist**: Add to antivirus exceptions
3. **Sign code**: Reduces false positives
4. **Report**: Submit to antivirus vendors as false positive

---

## 🗂️ File Structure After Build

```
Stock_price_prediction/
├── dist/
│   └── StockPredictor.exe      # ✅ Distributable executable
├── build/                       # ⚠️  Build artifacts (delete after successful build)
├── src/                         # Source code (not needed for distribution)
├── data/                        # Created at runtime
├── models/                      # Created at runtime
├── stock_predictor.spec         # Build configuration
└── requirements.txt             # For source builds
```

### What to Include in Distribution

**Minimum** (for executable):
```
StockPredictor.exe
README.md (usage instructions)
```

**Recommended**:
```
StockPredictor.exe
README.md
ACCURACY_GUIDE.md
LICENSE.txt
```

**Full Package** (for developers):
```
All source files
requirements.txt
Documentation
Tests
```

---

## 📊 Performance Optimization

### Reduce Executable Size

1. **Exclude unused libraries**:
   ```python
   # In stock_predictor.spec
   excludes=['matplotlib', 'tkinter']  # If not used
   ```

2. **Use UPX compression**:
   ```powershell
   # Download UPX from upx.github.io
   pyinstaller --upx-dir=C:\upx stock_predictor.spec
   ```

3. **Remove debug info**:
   ```python
   # In spec file
   debug=False
   console=False  # For GUI apps
   ```

### Improve Startup Time

1. **Use directory mode** instead of onefile
2. **Lazy imports**: Import libraries only when needed
3. **Cache models**: Don't retrain every time

---

## 🐛 Troubleshooting Build Issues

### Issue 1: "ModuleNotFoundError" in Executable

**Solution**: Add to `hiddenimports` in spec file
```python
hiddenimports=[
    'your.missing.module',
]
```

### Issue 2: "Failed to execute script"

**Solution**: Run in console mode to see errors
```python
# In spec file
console=True,
debug=True,
```

### Issue 3: "DLL load failed"

**Solution**: Include missing DLLs
```python
binaries=[
    ('path/to/missing.dll', '.'),
],
```

### Issue 4: Large File Size

**Solutions**:
1. Use virtual environment before building
2. Exclude unnecessary packages
3. Use directory mode distribution

### Issue 5: Slow Build Time

**Solutions**:
1. Disable antivirus temporarily
2. Use SSD for build directory
3. Close other applications

---

## ✅ Pre-Distribution Checklist

Before sharing your executable:

- [ ] Test on clean Windows machine (no Python installed)
- [ ] Test all command-line arguments
- [ ] Test interactive mode
- [ ] Verify model training works
- [ ] Check predictions are accurate
- [ ] Test with different stock symbols
- [ ] Verify error handling
- [ ] Check file permissions
- [ ] Test network connectivity requirements
- [ ] Create user documentation
- [ ] Add version number/about info
- [ ] Consider code signing
- [ ] Scan with antivirus
- [ ] Test on Windows 10 & 11

---

## 📝 Version Management

### Adding Version Info

Update `stock_predictor.spec`:

```python
exe = EXE(
    # ... existing config ...
    version='version_info.txt',  # Create this file
    name='StockPredictor_v1.0.0',
)
```

Create `version_info.txt`:
```
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable('040904B0', [
        StringStruct('CompanyName', 'Your Company'),
        StringStruct('FileDescription', 'Stock Price Prediction System'),
        StringStruct('FileVersion', '1.0.0.0'),
        StringStruct('ProductName', 'Stock Predictor'),
        StringStruct('ProductVersion', '1.0.0.0')])
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
```

---

## 🌐 Future Enhancements

### Planned Features

1. **Auto-Update System**:
   - Check for updates on startup
   - Download and install automatically

2. **GUI Version**:
   - Build with Tkinter/PyQt
   - More user-friendly interface

3. **Web API**:
   - REST API endpoint
   - Deploy to cloud (AWS/Azure/GCP)

4. **Docker Container**:
   - Cross-platform deployment
   - Easy cloud deployment

5. **Mobile App**:
   - React Native
   - Flutter

---

## 📞 Support & Resources

### Documentation
- [User Guide](README.md)
- [Accuracy Guide](ACCURACY_GUIDE.md)
- [Troubleshooting](TROUBLESHOOTING.md)

### Development
- **PyInstaller Docs**: https://pyinstaller.org/
- **Python Packaging**: https://packaging.python.org/
- **Inno Setup**: https://jrsoftware.org/isinfo.php

### Contact
- **Issues**: GitHub Issues
- **Email**: your.email@example.com
- **Forum**: Your support forum URL

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

**Build Date**: December 3, 2025  
**Version**: 1.0.0  
**Python Version**: 3.11.9  
**PyInstaller Version**: 5.13.0

**Status**: ✅ Production Ready
