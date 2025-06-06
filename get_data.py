
import json
import pandas as pd
import plotly.express as px
import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)

# Suppress pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None

def fetch_data(symbol):
    cutoff= 2016

    new_data=pd.read_json(f"{symbol}.json").T
    new_data.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'}, inplace=True)

    print(type(new_data))

    # Fix the column names to match the actual JSON structure
    # Convert to numeric first, then apply pandas operations
    # Convert all price columns to numeric
    new_data['open'] = pd.to_numeric(new_data['open'])
    new_data['high'] = pd.to_numeric(new_data['high'])
    new_data['low'] = pd.to_numeric(new_data['low'])
    new_data['close'] = pd.to_numeric(new_data['close'])

    # Convert index to datetime for proper time-based calculations
    new_data.index = pd.to_datetime(new_data.index)

    # Sort by date to ensure proper chronological order
    new_data = new_data.sort_index()

    # Existing calculations
    new_data["Lag"] = new_data['open'].shift(1)
    new_data['RollingMean'] = new_data['open'].rolling(window=30).mean()

    # 52-week range high - low over prior 252 trading days, approximately 1 year
    new_data['52_week_high'] = new_data['high'].rolling(window=252).max()
    new_data['52_week_low'] = new_data['low'].rolling(window=252).min()
    new_data['52_week_range'] = new_data['52_week_high'] - new_data['52_week_low']

    # 5-year volatility based on monthly averages
    # First calculate monthly averages
    monthly_avg = new_data['close'].resample('M').mean()
    new_data['Log_Return'] = np.log(new_data['open'] / new_data["Lag"])

    # Define the window for 3-year volatility number of trading days in 3 years
    window = 3 * 252

    # volatility when there is enough data for a full 3-year period.
    new_data['3_Year_Volatility'] = new_data['Log_Return'].rolling(window=window, min_periods=window).std() * np.sqrt(252)

    # Short-term volatility: 30-day spread (high - low over 30 days)
    window = 30
    annualization_factor = np.sqrt(252)
    new_data['short_term_volatility'] = new_data['Log_Return'].rolling(window=window, min_periods=window).std() * annualization_factor

    data=new_data[new_data.index > str(cutoff)]

    print(data.iloc[:,-7:])

    data.to_csv(f'{symbol}.csv', index=True, index_label='Date')
