# 📈 Stock Price Prediction System

A comprehensive machine learning project for predicting Indian stock market prices using ARIMA and LSTM models.

## 🎯 Project Overview

This project provides a complete MVP (Minimum Viable Product) for stock price prediction tailored for the Indian stock market. It fetches real-time and historical data from NSE (National Stock Exchange), engineers technical indicators, and uses both traditional statistical models (ARIMA) and deep learning (LSTM) for predictions.

## ✨ Features

- **Data Fetching**: Retrieve historical and live stock data from NSE using `nsepython`
- **Data Preprocessing**: Handle missing values, outliers, and normalize data
- **Technical Indicators**: 
  - Moving Averages (MA10, MA20, MA50)
  - Exponential Moving Averages (EMA)
  - Relative Strength Index (RSI)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Volume-based indicators
  - Price-based features
- **Dual Model Approach**:
  - ARIMA for time-series forecasting
  - LSTM neural network for deep learning prediction
- **Model Evaluation**: 
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
  - R-Squared (R²)
  - Directional Accuracy
- **CLI Interface**: Easy-to-use command-line interface
- **Modular Design**: Clean, production-ready code structure

## 📁 Project Structure

```
Stock_price_prediction/
│
├── src/
│   ├── config.py              # Configuration and constants
│   ├── data_fetcher.py        # Data fetching from NSE
│   ├── preprocess.py          # Data preprocessing and cleaning
│   ├── features.py            # Technical indicator engineering
│   ├── train_model.py         # Model training (ARIMA & LSTM)
│   ├── predict.py             # Price prediction module
│   ├── evaluate.py            # Model evaluation metrics
│   └── cli_interface.py       # Command-line interface
│
├── models/                    # Saved trained models
├── data/                      # Historical stock data (CSV)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or Download the Project**

```powershell
cd "c:\Users\Dhanarajan K\OneDrive\Desktop\Dhaannn\SPP\Stock_price_prediction"
```

2. **Create Virtual Environment (Recommended)**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install Dependencies**

```powershell
pip install -r requirements.txt
```

## 💻 Usage

### Interactive Mode

Launch the interactive CLI:

```powershell
cd src
python cli_interface.py -i
```

### Command-Line Mode

**Fetch data and train models:**

```powershell
cd src
python cli_interface.py RELIANCE --train --history 365
```

**Make predictions using existing models:**

```powershell
cd src
python cli_interface.py RELIANCE --days 5 --model both
```

**Full pipeline (fetch, train, predict):**

```powershell
cd src
python cli_interface.py TCS --train --days 5
```

### Command-Line Arguments

- `symbol`: Stock symbol (e.g., RELIANCE, TCS, INFY, HDFCBANK)
- `-d, --days`: Number of days to forecast (default: 5)
- `-m, --model`: Model to use - `arima`, `lstm`, or `both` (default: both)
- `-t, --train`: Train models before prediction
- `-i, --interactive`: Run in interactive mode
- `--history`: Days of historical data to fetch (default: 365)

### Example Stock Symbols

- **RELIANCE** - Reliance Industries
- **TCS** - Tata Consultancy Services
- **INFY** - Infosys
- **HDFCBANK** - HDFC Bank
- **ICICIBANK** - ICICI Bank
- **SBIN** - State Bank of India
- **WIPRO** - Wipro Limited
- **BHARTIARTL** - Bharti Airtel

## 🔧 Module Details

### 1. config.py
Central configuration file containing:
- Directory paths
- Model parameters (ARIMA order, LSTM architecture)
- Technical indicator settings
- Training hyperparameters

### 2. data_fetcher.py
Fetches stock data from NSE:
- Uses `nsepython` library for Indian stocks
- Fallback to sample data for testing
- Saves data to CSV for future use

### 3. preprocess.py
Data cleaning and normalization:
- Handles missing values
- Removes outliers
- MinMax scaling for neural networks
- Creates lagged and rolling features

### 4. features.py
Technical indicator engineering:
- Moving averages and EMAs
- RSI calculation
- MACD indicators
- Bollinger Bands
- Volume and price features

### 5. train_model.py
Model training:
- ARIMA with auto-tuning capability
- LSTM with customizable architecture
- Model saving for later use

### 6. predict.py
Prediction module:
- Loads trained models
- Makes multi-day forecasts
- Inverse transforms predictions to original scale

### 7. evaluate.py
Model evaluation:
- MAE, RMSE, MAPE metrics
- R² and directional accuracy
- Model comparison utilities

### 8. cli_interface.py
User-friendly interface:
- Interactive and command-line modes
- Full pipeline automation
- Clear output formatting

## 📊 Model Performance

### ARIMA Model
- **Strengths**: Good for short-term predictions, captures trends
- **Weaknesses**: Limited for non-linear patterns
- **Best For**: Stable stocks with clear trends

### LSTM Model
- **Strengths**: Captures complex patterns, learns from features
- **Weaknesses**: Requires more data, prone to overfitting
- **Best For**: Volatile stocks with complex patterns

### Ensemble Approach
The system provides predictions from both models, allowing you to:
- Compare model performance
- Use ensemble (average) for robust predictions
- Select best model based on evaluation metrics

## 🎓 Technical Details

### ARIMA Parameters
- Default Order: (5, 1, 0)
- Auto-tuning available using `pmdarima`
- Suitable for univariate time series

### LSTM Architecture
- Input: 60-day sequences
- Two LSTM layers (50 units each)
- Dropout: 0.2 for regularization
- Optimizer: Adam (learning rate: 0.001)
- Early stopping to prevent overfitting

### Data Split
- Training: 80%
- Testing: 20%
- Validation: 20% of training data

## 📦 Creating Executable

To create a standalone `.exe` file:

```powershell
cd src
pyinstaller --onefile --name StockPredictor cli_interface.py
```

The executable will be in the `dist/` folder.

## ⚠️ Important Notes

1. **Data Availability**: NSE data may have delays or restrictions. The system includes fallback sample data for testing.

2. **Minimum Data Requirements**:
   - **Absolute Minimum**: 100 days of historical data
   - **Recommended**: 365+ days (1 year) for reliable predictions
   - **Best Results**: 730+ days (2 years) of data
   - Technical indicators like MA50 require sufficient historical windows

3. **Internet Connection**: Required for fetching live stock data.

4. **Model Training Time**: LSTM training can take 5-30 minutes depending on data size and hardware.

5. **Disclaimer**: This is for educational and research purposes. Do not use for actual trading without proper validation and risk management.

## 🔍 Troubleshooting

### Issue: "nsepython not found"
```powershell
pip install nsepython
```

### Issue: "TensorFlow import error"
```powershell
pip install tensorflow==2.13.0
```

### Issue: "Models not found"
Run with `--train` flag first:
```powershell
python cli_interface.py RELIANCE --train
```

### Issue: "No data fetched"
Check internet connection or use the fallback data generation feature.

## 📝 Example Workflow

1. **First-time setup:**
```powershell
# Fetch data and train models for RELIANCE
python cli_interface.py RELIANCE --train --history 730
```

2. **Daily predictions:**
```powershell
# Get 5-day forecast
python cli_interface.py RELIANCE --days 5
```

3. **Model comparison:**
```powershell
# Compare ARIMA vs LSTM
python cli_interface.py RELIANCE --model both --days 10
```

## 🎯 Interview Presentation Tips

1. **Demonstrate End-to-End Pipeline**: Show data fetching → preprocessing → training → prediction
2. **Explain Technical Indicators**: Discuss why RSI, MACD, etc. are important
3. **Compare Models**: Show ARIMA vs LSTM performance on sample stock
4. **Show Modularity**: Explain how each module can be independently tested
5. **Discuss Scalability**: Mention how to extend to multiple stocks, real-time trading

## 📚 Future Enhancements

- [ ] Add more models (GRU, Transformer)
- [ ] Implement real-time predictions
- [ ] Add web interface (Flask/Streamlit)
- [ ] Multi-stock portfolio prediction
- [ ] Sentiment analysis integration
- [ ] Advanced ensemble methods

## 🤝 Contributing

This is an educational project. Feel free to fork, modify, and enhance!

## 📄 License

This project is provided as-is for educational purposes.

## 👨‍💻 Author

Dhanarajan K

---

**Good luck with your interview! 🚀**
