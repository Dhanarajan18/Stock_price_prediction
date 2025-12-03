# 🎯 Complete Installation & Usage Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [First Time Setup](#first-time-setup)
4. [Usage Examples](#usage-examples)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Usage](#advanced-usage)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8 or higher
- **RAM**: 4 GB (8 GB recommended)
- **Disk Space**: 5 GB free space
- **Internet**: Required for downloading stock data

### Recommended Specifications
- **Python**: 3.10 or 3.11
- **RAM**: 8 GB or more
- **GPU**: NVIDIA GPU with CUDA (optional, for faster LSTM training)
- **Internet**: Broadband connection

---

## Installation Methods

### Method 1: Automated Setup (Recommended)

```powershell
# Navigate to project directory
cd "c:\Users\Dhanarajan K\OneDrive\Desktop\Dhaannn\SPP\Stock_price_prediction"

# Run setup script
python setup.py
```

This will:
- Check Python version
- Install all dependencies
- Create necessary directories
- Verify installation
- Test all modules

### Method 2: Manual Installation

#### Step 1: Create Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get an error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Step 2: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

**Note**: This will download ~2-3 GB of packages and may take 10-15 minutes.

#### Step 3: Verify Installation

```powershell
# Test import
python -c "import numpy, pandas, tensorflow, sklearn; print('All packages installed!')"
```

---

## First Time Setup

### 1. Understand the Project Structure

```
Stock_price_prediction/
├── src/              # All source code
├── data/             # Stock data (created automatically)
├── models/           # Trained models (created automatically)
├── demo.py           # Full demonstration
├── main.py           # Main entry point
├── setup.py          # Setup script
└── requirements.txt  # Dependencies
```

### 2. Test with Demo

```powershell
# Run the full demo
python demo.py
```

This will:
- Fetch sample data
- Create features
- Train both models
- Make predictions
- Show evaluation metrics

**Time**: ~10-30 minutes (depending on your hardware)

### 3. Try Interactive Mode

```powershell
# Start interactive CLI
python src/cli_interface.py -i
```

Then follow the on-screen menu.

---

## Usage Examples

### Example 1: Quick Prediction (Requires Pre-trained Models)

```powershell
# Make 5-day forecast for RELIANCE
python src/cli_interface.py RELIANCE --days 5

# Make 10-day forecast using only LSTM
python src/cli_interface.py TCS --days 10 --model lstm
```

### Example 2: Train Models First

```powershell
# Train models with 2 years of data
python src/cli_interface.py INFY --train --history 730

# Train and immediately predict
python src/cli_interface.py HDFCBANK --train --days 5
```

### Example 3: Full Pipeline

```powershell
# Complete workflow: fetch → train → predict
python main.py RELIANCE --train --history 1095 --days 7 --model both
```

### Example 4: Batch Processing Multiple Stocks

Create a PowerShell script `batch_predict.ps1`:

```powershell
$stocks = @('RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK')

foreach ($stock in $stocks) {
    Write-Host "Processing $stock..."
    python src/cli_interface.py $stock --train --days 5
    Write-Host "Completed $stock`n"
}
```

Run with:
```powershell
.\batch_predict.ps1
```

---

## Usage Scenarios

### Scenario 1: Interview Presentation

**Preparation** (Do this before the interview):
```powershell
# Train models for 2-3 popular stocks
python src/cli_interface.py RELIANCE --train --history 730
python src/cli_interface.py TCS --train --history 730
python src/cli_interface.py INFY --train --history 730
```

**During Interview**:
```powershell
# Show quick predictions
python src/cli_interface.py RELIANCE --days 5
python src/cli_interface.py TCS --days 5

# Or run the demo
python demo.py
```

### Scenario 2: Daily Stock Analysis

```powershell
# Update data and get fresh predictions
python src/cli_interface.py RELIANCE --days 5
python src/cli_interface.py HDFCBANK --days 5
```

### Scenario 3: Model Comparison

```powershell
# Compare ARIMA vs LSTM
python src/cli_interface.py SBIN --model arima --days 5
python src/cli_interface.py SBIN --model lstm --days 5
python src/cli_interface.py SBIN --model both --days 5
```

---

## Command-Line Reference

### Main Commands

```powershell
# Interactive mode
python src/cli_interface.py -i

# Direct prediction
python src/cli_interface.py <SYMBOL> [OPTIONS]

# Run demo
python demo.py

# Setup
python setup.py
```

### Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `-d, --days` | Forecast days | 5 |
| `-m, --model` | Model type (arima/lstm/both) | both |
| `-t, --train` | Train models before prediction | False |
| `-i, --interactive` | Interactive mode | False |
| `--history` | Days of historical data | 365 |

### Examples

```powershell
# Get 3-day forecast
python src/cli_interface.py WIPRO --days 3

# Use only ARIMA model
python src/cli_interface.py BHARTIARTL --model arima

# Train with 3 years of data
python src/cli_interface.py TATAMOTORS --train --history 1095

# 10-day LSTM forecast
python src/cli_interface.py SUNPHARMA --days 10 --model lstm
```

---

## Troubleshooting

### Issue 1: Import Errors

**Problem**:
```
ImportError: No module named 'numpy'
```

**Solution**:
```powershell
pip install -r requirements.txt
```

### Issue 2: TensorFlow Not Found

**Problem**:
```
Import "tensorflow" could not be resolved
```

**Solution**:
```powershell
pip install tensorflow==2.13.0
```

### Issue 3: NSE Data Not Available

**Problem**:
```
Failed to fetch data for RELIANCE
```

**Solutions**:
1. Check internet connection
2. The system will automatically use sample data
3. Try a different stock symbol
4. Install nsepython: `pip install nsepython`

### Issue 4: Models Not Found

**Problem**:
```
ARIMA model not found
```

**Solution**:
```powershell
# Train models first
python src/cli_interface.py RELIANCE --train
```

### Issue 5: Memory Error During Training

**Problem**:
```
MemoryError: Unable to allocate array
```

**Solutions**:
1. Close other applications
2. Reduce `--history` parameter
3. Use smaller LSTM architecture (edit `src/config.py`)

### Issue 6: Slow LSTM Training

**Problem**: LSTM takes too long to train

**Solutions**:
1. Reduce `LSTM_EPOCHS` in `src/config.py` (try 25 instead of 50)
2. Use GPU if available
3. Reduce `--history` parameter
4. Train once, reuse models

### Issue 7: PowerShell Script Execution Error

**Problem**:
```
cannot be loaded because running scripts is disabled
```

**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Advanced Usage

### Customizing Model Parameters

Edit `src/config.py`:

```python
# Increase forecast days
FORECAST_DAYS = 10

# Change LSTM architecture
LSTM_UNITS = [100, 100]  # Larger network
LSTM_EPOCHS = 100        # More training

# Adjust ARIMA order
ARIMA_ORDER = (7, 1, 1)  # Different parameters
```

### Using Different Technical Indicators

Edit `src/config.py`:

```python
# Add more moving averages
MOVING_AVERAGES = [5, 10, 20, 50, 100, 200]

# Change RSI period
RSI_PERIOD = 21

# Adjust Bollinger Bands
BOLLINGER_PERIOD = 30
BOLLINGER_STD = 3
```

### Creating Executable

```powershell
# Install PyInstaller
pip install pyinstaller

# Create executable
pyinstaller stock_predictor.spec

# Executable will be in dist/ folder
.\dist\StockPredictor.exe RELIANCE --days 5
```

### Integrating with Other Tools

**Use as Python Module**:

```python
import sys
sys.path.insert(0, 'src')

from data_fetcher import fetch_stock_data
from train_model import train_models
from predict import make_predictions

# Fetch data
df = fetch_stock_data('RELIANCE', days=365)

# Train models
arima_model, lstm_model = train_models(df)

# Make predictions
predictions = make_predictions(df['Close'].values, steps=5)
```

---

## Performance Optimization

### For Faster Training

1. **Use GPU**:
   - Install CUDA and cuDNN
   - Install tensorflow-gpu

2. **Reduce Data**:
   - Use `--history 365` instead of 1095

3. **Reduce Epochs**:
   - Change `LSTM_EPOCHS` to 25-30

4. **Parallel Processing**:
   - Train multiple stocks in separate terminals

### For Better Predictions

1. **More Data**:
   - Use `--history 1095` (3 years)

2. **Auto-tune ARIMA**:
   - Edit `train_model.py`, set `auto_tune=True`

3. **Ensemble Methods**:
   - Average ARIMA and LSTM predictions

4. **Feature Selection**:
   - Experiment with different technical indicators

---

## Best Practices

### For Development
- ✅ Use virtual environment
- ✅ Version control with Git
- ✅ Test with sample data first
- ✅ Keep models directory organized
- ✅ Comment your modifications

### For Production Use
- ✅ Validate all predictions
- ✅ Implement risk management
- ✅ Monitor model performance
- ✅ Retrain models regularly
- ✅ Keep logs of predictions

### For Interview/Demo
- ✅ Pre-train models beforehand
- ✅ Test everything before presentation
- ✅ Have backup (sample data)
- ✅ Prepare talking points
- ✅ Understand the code thoroughly

---

## Resources

### Documentation
- `README.md` - Complete project documentation
- `QUICKSTART.md` - Quick start guide
- `PROJECT_SUMMARY.md` - Project overview
- Code comments and docstrings

### Support
- Check error messages carefully
- Review troubleshooting section
- Test with sample data
- Verify dependencies installed

### Popular Stock Symbols

**Banking**:
- HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK

**IT**:
- TCS, INFY, WIPRO, HCLTECH, TECHM

**Energy**:
- RELIANCE, ONGC, BPCL, IOC

**Auto**:
- TATAMOTORS, MARUTI, M&M, BAJAJ-AUTO

**Pharma**:
- SUNPHARMA, DRREDDY, CIPLA, DIVISLAB

---

## Next Steps

1. **Complete Setup**:
   ```powershell
   python setup.py
   ```

2. **Run Demo**:
   ```powershell
   python demo.py
   ```

3. **Try Predictions**:
   ```powershell
   python src/cli_interface.py RELIANCE --train --days 5
   ```

4. **Read Documentation**:
   - Start with README.md
   - Review code in src/
   - Experiment with parameters

5. **Customize**:
   - Modify config.py
   - Try different stocks
   - Adjust model parameters

---

**You're all set! Happy predicting! 📈🚀**

*Last Updated: December 3, 2025*
