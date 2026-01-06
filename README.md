# Pine Script ML Indicator Optimizer

A machine learning-powered optimization tool for Pine Script indicators. Uses Bayesian optimization (Optuna TPE) with walk-forward validation to find optimal indicator parameters that maximize profitability and directional accuracy.

## Features

- **Bayesian Optimization**: Uses Tree-Parzen Estimator (TPE) for efficient parameter search
- **Walk-Forward Validation**: 5-fold cross-validation with embargo periods to prevent look-ahead bias
- **Multi-Symbol Testing**: Tests indicators across multiple cryptocurrency pairs (BTC, ETH, SOL, XRP, ADA)
- **Automatic Data Management**: Downloads and caches historical 1H OHLCV data from Binance
- **Performance Metrics**: Evaluates profit factor, win rate, Sharpe ratio, and directional accuracy
- **Forecast Horizon Detection**: Automatically finds optimal holding period for signals

## Prerequisites

- Python 3.8 or higher
- Internet connection (for downloading historical data from Binance)

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

The required packages are:
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `optuna>=3.4.0`
- `requests>=2.31.0`

## Quick Start

### Basic Usage

Optimize a single Pine Script indicator:

```bash
python optimize_indicator.py <pine_script_file>
```

Example:
```bash
python optimize_indicator.py MACD_demo_pinescript.pine
```

### Optimize All Indicators

If you run the script without arguments, it will automatically find and optimize all `.pine` files in the current directory (excluding already optimized files):

```bash
python optimize_indicator.py
```

## Command-Line Options

```bash
python optimize_indicator.py <pine_script_file> [options]
```

**Options:**

- `--max-trials N` - Maximum optimization trials (default: 150)
  - More trials = better results but longer runtime
  - Recommended: 100-200 for quick tests, 300+ for production

- `--timeout N` - Maximum time in seconds (default: 300 = 5 minutes)
  - Optimization will stop after this time even if max-trials not reached

- `--symbols LIST` - Comma-separated symbols to test (default: all)
  - Available symbols: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, ADAUSDT
  - Example: `--symbols BTCUSDT,ETHUSDT`

- `--force-download` - Force re-download of historical data
  - Use this if you want fresh data or suspect data corruption

- `--verbose` - Enable verbose logging
  - Shows detailed debug information during optimization

## Examples

### Quick Test (Fewer Trials)
```bash
python optimize_indicator.py EyeX_MFV_v5.pine --max-trials 50 --timeout 120
```

### Production Run (More Trials)
```bash
python optimize_indicator.py pso_indicator.pine --max-trials 300 --timeout 600
```

### Test on Specific Symbols Only
```bash
python optimize_indicator.py MACD_demo_pinescript.pine --symbols BTCUSDT,ETHUSDT
```

### Verbose Output
```bash
python optimize_indicator.py indicator.pine --verbose
```

## How It Works

1. **Data Loading**: Downloads or loads historical 1H OHLCV data from Binance for specified symbols
2. **Pine Script Parsing**: Extracts parameters, signal conditions, and indicator logic
3. **Optimization**: Uses Optuna TPE sampler to search parameter space
4. **Backtesting**: Evaluates each parameter set using walk-forward validation
5. **Output Generation**: Creates optimized Pine Script and performance report

### Optimization Process

- **Walk-Forward Validation**: Splits data into 5 folds with embargo periods
- **Objective Function**: Maximizes weighted combination of:
  - Profit Factor (35% weight)
  - Directional Accuracy (30% weight)
  - Sharpe Ratio (20% weight)
  - Win Rate (15% weight)
- **Early Pruning**: Automatically discards poor parameter configurations
- **Forecast Horizon**: Tests multiple time horizons (6h, 12h, 24h, 48h, 72h, 96h) to find optimal holding period

## Output Files

After optimization completes, two files are generated:

1. **`optimised_<original_filename>.pine`** - The optimized Pine Script with best parameters
2. **`optimised_<original_filename>_report.txt`** - Detailed performance report

### Report Contents

The report includes:
- Performance metrics (profit factor, win rate, Sharpe ratio)
- Directional accuracy vs random baseline
- Optimal forecast horizon
- Parameter changes from original
- Performance comparison (original vs optimized)

## Data Management

Historical data is stored in the `historical_data/` directory as CSV files:
- Format: `{SYMBOL}_1h.csv`
- Columns: timestamp, open, high, low, close, volume
- Data is cached locally - subsequent runs use cached data unless `--force-download` is used

## Pine Script Requirements

Your Pine Script indicator should:

1. Use `@version=5` or `@version=6` (v6 recommended)
2. Have optimizable parameters defined with `input.int()`, `input.float()`, or `input.bool()`
3. Include buy/sell signals (via `plotshape()`, `plotchar()`, or conditional logic)
4. Follow Pine Script v6 syntax conventions

### Example Indicator Structure

```pine
//@version=6
indicator("My Indicator", overlay=false)

// Optimizable parameters
length = input.int(14, "Length", minval=1, maxval=100)
threshold = input.float(0.5, "Threshold", minval=0.0, maxval=1.0)

// Indicator calculation
value = ta.sma(close, length)

// Signals
buySignal = value > threshold
sellSignal = value < -threshold

// Plotting
plot(value, "Value")
plotshape(buySignal, "Buy", shape.triangleup, location.belowbar, color.green)
plotshape(sellSignal, "Sell", shape.triangledown, location.abovebar, color.red)
```

## Troubleshooting

### No Data Available
If you see "No historical data available":
- Check your internet connection
- Verify symbol names are correct (must be uppercase, e.g., `BTCUSDT`)
- Try `--force-download` to re-download data

### Optimization Takes Too Long
- Reduce `--max-trials` (e.g., 50-100)
- Reduce `--timeout` (e.g., 120 seconds)
- Test on fewer symbols with `--symbols`

### Poor Optimization Results
- Increase `--max-trials` for better search
- Check that your indicator has meaningful buy/sell signals
- Verify parameter ranges are appropriate (not too narrow or too wide)

### Pine Script Parsing Errors
- Ensure your script uses valid Pine Script v5/v6 syntax
- Check that parameters use `input.int()`, `input.float()`, or `input.bool()`
- Verify signal conditions are properly defined

## Project Structure

```
pinescript-ml-optimisation/
├── optimize_indicator.py    # Main entry point
├── data_manager.py          # Historical data download/management
├── pine_parser.py           # Pine Script parsing
├── pine_translator.py       # Pine Script to Python translation
├── optimizer.py             # Optuna optimization logic
├── backtester.py            # Walk-forward backtesting
├── output_generator.py      # Generate optimized scripts and reports
├── ta_functions.py          # Technical analysis functions
├── requirements.txt         # Python dependencies
├── historical_data/         # Cached CSV data
│   ├── BTCUSDT_1h.csv
│   ├── ETHUSDT_1h.csv
│   └── ...
└── *.pine                   # Your Pine Script indicators
```

## Performance Tips

- **First Run**: Allow extra time for data download (5-10 minutes depending on connection)
- **Subsequent Runs**: Much faster as data is cached locally
- **Resource Usage**: Optimization is CPU-intensive; close other applications for best performance
- **Trial Count**: Start with 50-100 trials to test, then increase to 200-300 for production

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions:
1. Check that your Pine Script follows v5/v6 syntax
2. Verify all dependencies are installed correctly
3. Review the verbose output with `--verbose` flag
4. Check that historical data files exist in `historical_data/` directory

