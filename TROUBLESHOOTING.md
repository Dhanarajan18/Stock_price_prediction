# ⚠️ Common Issues & Solutions

## Issue: "No data remaining after feature engineering"

### Problem
```
❌ Error: No data remaining after feature engineering!
   Technical indicators require historical windows (e.g., MA50 needs 50 days).
```

### Cause
You're trying to use too little historical data. Technical indicators like:
- **MA50** (50-day moving average) needs at least 50 days
- **Bollinger Bands** (20-period) needs at least 20 days
- After calculating all indicators, rows with incomplete data are dropped

With only 15 days of data, after applying all indicators, no usable data remains.

### Solution
**Use more historical data:**

```powershell
# Minimum (may still be insufficient)
python src/cli_interface.py RELIANCE --train --history 100

# Recommended minimum
python src/cli_interface.py RELIANCE --train --history 365

# Best for predictions
python src/cli_interface.py RELIANCE --train --history 730
```

### Data Requirements

| Purpose | Minimum Days | Recommended Days |
|---------|-------------|------------------|
| Testing | 100 | 200 |
| Training | 200 | 365 |
| Good Predictions | 365 | 730+ |
| Best Results | 730 | 1095+ |

---

## Issue: "ValueError: Found array with 0 sample(s)"

### Problem
```
ValueError: Found array with 0 sample(s) (shape=(0, 5)) while a minimum of 1 is required by MinMaxScaler.
```

### Cause
Same as above - insufficient data after feature engineering.

### Solution
Increase the `--history` parameter:

```powershell
# Interactive mode
python src/cli_interface.py -i
# Then enter: 365 (or more) when asked for days of history

# Command-line mode
python src/cli_interface.py RELIANCE --train --history 365
```

---

## Issue: "Insufficient data for LSTM"

### Problem
```
⚠️  Warning: Only 45 records after feature engineering.
   LSTM model requires at least 60 records.
```

### Cause
LSTM uses sequences of 60 days to make predictions. You need at least 60 data points after feature engineering.

### Solution
Fetch more historical data:

```powershell
# Get 1 year of data (recommended)
python src/cli_interface.py RELIANCE --train --history 365

# Get 2 years for better results
python src/cli_interface.py RELIANCE --train --history 730
```

---

## Issue: NSE Data Fetch Failed

### Problem
```
WARNING - NSE fetch failed: Expecting value: line 1 column 1 (char 0)
WARNING - No data received for RELIANCE. Trying fallback method.
```

### Cause
1. Internet connection issue
2. NSE API temporarily unavailable
3. Invalid stock symbol
4. Market closed/holidays

### Solution

**Option 1: Use fallback sample data** (for testing)
```powershell
# The system automatically generates sample data
# This is fine for testing but NOT real market data
```

**Option 2: Check internet connection**
```powershell
# Test connection
ping www.nseindia.com
```

**Option 3: Try different stock symbol**
```powershell
# Valid symbols:
python src/cli_interface.py TCS --train --history 365
python src/cli_interface.py INFY --train --history 365
python src/cli_interface.py HDFCBANK --train --history 365
```

**Option 4: Try later**
- NSE might be experiencing temporary issues
- Try during market hours (9:15 AM - 3:30 PM IST)

---

## Quick Reference: Historical Data Guidelines

### For Interactive Mode
When asked "Enter number of days of history":
- **Minimum**: 100
- **Good**: 365
- **Better**: 730
- **Best**: 1095

### For Command-Line Mode
```powershell
# Minimum (100 days)
--history 100

# Recommended (1 year)
--history 365

# Better (2 years)
--history 730

# Best (3 years)
--history 1095
```

---

## How Much Data Do I Really Need?

### Why More Data is Better

1. **Technical Indicators**: Need historical windows
   - MA10 needs 10 days
   - MA20 needs 20 days
   - MA50 needs 50 days
   - After calculation, early rows are dropped

2. **Model Training**: Need enough samples
   - ARIMA: Minimum 30 samples
   - LSTM: Minimum 60 samples (sequences)
   - More data = better learning

3. **Train-Test Split**: 80/20 split
   - 100 days → 80 train, 20 test (too little!)
   - 365 days → 292 train, 73 test (better)
   - 730 days → 584 train, 146 test (good!)

### Calculation Example

Starting with **15 days**:
1. After cleaning: 12 days
2. After feature engineering (MA50, etc.): **0 days** ❌
3. Result: Cannot train

Starting with **365 days**:
1. After cleaning: ~360 days
2. After feature engineering: ~310 days ✅
3. Train/test split: 248/62
4. Result: Can train both models ✅

---

## Best Practices

### ✅ DO
- Use at least 365 days of historical data
- Check warnings about data sufficiency
- Start with popular stocks (RELIANCE, TCS, INFY)
- Test with demo.py first

### ❌ DON'T
- Use less than 100 days of data
- Ignore warnings about insufficient data
- Expect good predictions with minimal data
- Train models with sample/fallback data for real trading

---

## Testing the System

### Quick Test (Minimum Data)
```powershell
# This will work but predictions won't be great
python src/cli_interface.py RELIANCE --train --history 100 --days 3
```

### Recommended Test
```powershell
# Good balance of time and accuracy
python src/cli_interface.py RELIANCE --train --history 365 --days 5
```

### Best Test
```powershell
# Best predictions (takes longer to train)
python src/cli_interface.py RELIANCE --train --history 730 --days 5
```

---

## Updated Usage Examples

### Interactive Mode (Corrected)
```powershell
python src/cli_interface.py -i

# When prompted:
Enter stock symbol: RELIANCE
Enter number of days of history: 365  # NOT 15!
```

### Command-Line Mode (Recommended)
```powershell
# Train with 1 year of data
python src/cli_interface.py RELIANCE --train --history 365 --days 5

# Train with 2 years of data (better)
python src/cli_interface.py TCS --train --history 730 --days 5

# Train with 3 years of data (best)
python src/cli_interface.py INFY --train --history 1095 --days 7
```

---

## Summary

**The key takeaway**: 
📊 **Always use at least 365 days of historical data for meaningful predictions!**

Minimum: 100 days
Recommended: 365 days
Best: 730+ days

---

*This guide is part of the Stock Price Prediction System*
*Last Updated: December 3, 2025*
