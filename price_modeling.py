import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
import warnings
from datetime import timedelta
warnings.simplefilter(action='ignore', category=FutureWarning)

def forecast_stock_price(symbol, cutoff=2016, forecast_days=30):
    
    # Suppress pandas warnings
    pd.options.mode.chained_assignment = None

    try:
        # Load and process data
        data = pd.read_json(f"{symbol}.json").T
        data.rename(columns={
            '1. open': 'open', 
            '2. high': 'high', 
            '3. low': 'low', 
            '4. close': 'close', 
            '5. volume': 'volume'
        }, inplace=True)

        # Filter data by cutoff year
        new_data = data[data.index > str(cutoff)]

        # Convert to numeric
        for col in ['open', 'high', 'low', 'close']:
            new_data[col] = pd.to_numeric(new_data[col])

        # Convert index to datetime and sort
        new_data.index = pd.to_datetime(new_data.index)
        new_data = new_data.sort_index()

        # Calculate more realistic features
        new_data['log_price'] = np.log(new_data['open'])  # Use log prices for more stable modeling
        new_data['price_change'] = new_data['open'].pct_change()  # Daily returns
        new_data['volatility'] = new_data['price_change'].rolling(window=20).std()  # 20-day volatility
        new_data['rsi'] = calculate_rsi(new_data['open'], window=14)  # RSI indicator
        new_data['ma_20'] = new_data['open'].rolling(window=20).mean()
        new_data['ma_50'] = new_data['open'].rolling(window=50).mean()
        new_data['price_vs_ma20'] = (new_data['open'] - new_data['ma_20']) / new_data['ma_20']  # Price relative to MA
        
        # Remove NaN values
        clean_data = new_data.dropna()

        if len(clean_data) < 100:
            raise ValueError(f"Insufficient data for {symbol}. Need at least 100 data points.")
        
        # Check for data quality issues
        if clean_data['open'].isna().any():
            raise ValueError(f"Data contains NaN values for {symbol}")
        
        if (clean_data['open'] <= 0).any():
            raise ValueError(f"Data contains non-positive prices for {symbol}")
        
        # Check if we have recent data
        days_since_last = (pd.Timestamp.now() - clean_data.index[-1]).days
        if days_since_last > 30:
            print(f"Warning: Last data point is {days_since_last} days old")
        
        print(f"Using {len(clean_data)} data points from {clean_data.index[0].date()} to {clean_data.index[-1].date()}")

        # Calculate historical volatility for realistic bounds
        historical_vol = clean_data['price_change'].std() * np.sqrt(252)  # Annualized volatility
        daily_vol = historical_vol / np.sqrt(252)  # Daily volatility
        
        # Define realistic exogenous features
        exog_features = ['volatility', 'rsi', 'price_vs_ma20']

        # Try different model configurations for robustness
        model_configs = [
            # Simple ARIMA without seasonality
            {'order': (1, 1, 1), 'seasonal_order': None},
            # ARIMA with exogenous variables
            {'order': (2, 1, 2), 'seasonal_order': None},
            # SARIMA with weekly seasonality
            {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 5)},
            # Simple AR model
            {'order': (2, 0, 0), 'seasonal_order': None}
        ]
        
        model_fit = None
        successful_config = None
        
        for i, config in enumerate(model_configs):
            try:
                print(f"Trying model configuration {i+1}/{len(model_configs)}")
                
                if config['seasonal_order'] is None:
                    model = SARIMAX(
                        clean_data['log_price'], 
                        order=config['order'],
                        exog=clean_data[exog_features],
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                else:
                    model = SARIMAX(
                        clean_data['log_price'], 
                        order=config['order'],
                        seasonal_order=config['seasonal_order'],
                        exog=clean_data[exog_features],
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                
                model_fit = model.fit(
                    disp=False, 
                    maxiter=300,
                    method='lbfgs',
                    optim_score='approx'
                )
                
                successful_config = config
                print(f"Successfully fitted model with configuration: {config}")
                break
                
            except Exception as e:
                print(f"Model configuration {i+1} failed: {str(e)}")
                continue
        
        if model_fit is None:
            # Fallback to simple linear trend if all models fail
            print("All SARIMAX models failed, using simple linear trend forecast")
            return simple_trend_forecast(clean_data, symbol, forecast_days)

        current_price = clean_data['open'].iloc[-1]
        current_log_price = clean_data['log_price'].iloc[-1]

        # Create future dates first
        last_date = clean_data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        
        # Create realistic future exogenous variables with proper datetime index
        future_exog = pd.DataFrame(index=forecast_dates)
        
        # Mean-reverting volatility
        last_volatility = clean_data['volatility'].iloc[-1]
        long_term_vol = clean_data['volatility'].mean()
        
        # RSI tends to revert to 50
        last_rsi = clean_data['rsi'].iloc[-1]
        
        # Price vs MA20 tends to revert to 0
        last_price_vs_ma = clean_data['price_vs_ma20'].iloc[-1]
        
        # Generate mean-reverting exogenous features
        vol_reversion_speed = 0.1
        rsi_reversion_speed = 0.05
        price_ma_reversion_speed = 0.03
        
        for i in range(forecast_days):
            current_date = forecast_dates[i]
            # Mean-reverting volatility
            vol_shock = np.random.normal(0, daily_vol * 0.1)
            if i == 0:
                future_exog.loc[current_date, 'volatility'] = last_volatility + vol_reversion_speed * (long_term_vol - last_volatility) + vol_shock
            else:
                prev_date = forecast_dates[i-1]
                prev_vol = future_exog.loc[prev_date, 'volatility']
                future_exog.loc[current_date, 'volatility'] = prev_vol + vol_reversion_speed * (long_term_vol - prev_vol) + vol_shock
            # Mean-reverting RSI
            if i == 0:
                future_exog.loc[current_date, 'rsi'] = last_rsi + rsi_reversion_speed * (50 - last_rsi) + np.random.normal(0, 2)
            else:
                prev_date = forecast_dates[i-1]
                prev_rsi = future_exog.loc[prev_date, 'rsi']
                future_exog.loc[current_date, 'rsi'] = prev_rsi + rsi_reversion_speed * (50 - prev_rsi) + np.random.normal(0, 2)
            
            # Mean-reverting price vs MA20
            if i == 0:
                future_exog.loc[current_date, 'price_vs_ma20'] = last_price_vs_ma + price_ma_reversion_speed * (0 - last_price_vs_ma) + np.random.normal(0, 0.01)
            else:
                prev_date = forecast_dates[i-1]
                prev_price_ma = future_exog.loc[prev_date, 'price_vs_ma20']
                future_exog.loc[current_date, 'price_vs_ma20'] = prev_price_ma + price_ma_reversion_speed * (0 - prev_price_ma) + np.random.normal(0, 0.01)
        # Ensure to include future_exog when forecasting
        try:
            print(future_exog)
            log_forecast = model_fit.forecast(steps=forecast_days, exog=future_exog)
        except Exception as forecast_error:
            print(f"Forecast with exog failed: {forecast_error}")
            # Try forecast without exogenous variables as fallback
            log_forecast = model_fit.forecast(steps=forecast_days)
        
        # Apply realistic constraints to prevent extreme movements
        max_daily_change = daily_vol * 2  # Maximum 2-sigma daily move
        constrained_log_forecast = []
        
        prev_log_price = current_log_price
        for i in range(forecast_days):
            predicted_log_price = log_forecast.iloc[i]
            log_change = predicted_log_price - prev_log_price
            
            # Constrain daily log changes
            if abs(log_change) > max_daily_change:
                log_change = np.sign(log_change) * max_daily_change
            
            # Add some noise but keep it realistic
            log_change += np.random.normal(0, daily_vol * 0.5)
            
            new_log_price = prev_log_price + log_change
            constrained_log_forecast.append(new_log_price)
            prev_log_price = new_log_price
        
        # Convert back to prices
        forecast = np.exp(constrained_log_forecast)
        
        # Apply additional constraint: max 20% total change over forecast period
        total_change = (forecast[-1] - current_price) / current_price
        if abs(total_change) > 0.20:  # 20% max change
            scaling_factor = 0.20 / abs(total_change)
            forecast = current_price + (forecast - current_price) * scaling_factor

        forecast = pd.Series(forecast)

        # Create plot
        fig = go.Figure()

        # Historical data (last 200 days for better visualization)
        recent_data = clean_data.tail(200)
        fig.add_trace(go.Scatter(
            x=recent_data.index, 
            y=recent_data['open'],
            mode='lines', 
            name=f'Historical {symbol} Opening Price',
            line=dict(color='blue')
        ))

        # Forecasted prices
        fig.add_trace(go.Scatter(
            x=forecast_dates, 
            y=forecast,
            mode='lines', 
            name=f'Forecasted {symbol} Price',
            line=dict(color='red', dash='dash')
        ))

        # Add confidence bands
        uncertainty = daily_vol * np.sqrt(np.arange(1, forecast_days + 1)) * current_price
        upper_bound = forecast + uncertainty
        lower_bound = forecast - uncertainty
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=upper_bound,
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=lower_bound,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Confidence Band',
            fillcolor='rgba(255,0,0,0.2)'
        ))

        fig.update_layout(
            title=f'{symbol} Stock Price Forecast ({forecast_days} Days) - Constrained Model',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            showlegend=True
        )

        # Return results
        results = {
            'symbol': symbol,
            'current_price': current_price,
            'forecast': forecast.tolist(),
            'forecast_dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
            'forecast_min': forecast.min(),
            'forecast_max': forecast.max(),
            'forecast_mean': forecast.mean(),
            'figure': fig,
            'model_summary': str(model_fit.summary()),
            'total_change_percent': ((forecast[-1] - current_price) / current_price * 100),
            'daily_volatility': daily_vol * 100,
            'annualized_volatility': historical_vol * 100
        }

        return results

    except FileNotFoundError:
        raise FileNotFoundError(f"Data file {symbol}.json not found. Please ensure the data exists.")
    except Exception as e:
        raise Exception(f"Error forecasting {symbol}: {str(e)}")

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def simple_trend_forecast(clean_data, symbol, forecast_days):
    from sklearn.linear_model import LinearRegression
    
    # Use recent data for trend calculation
    recent_data = clean_data.tail(60)  # Last 60 days
    
    # Create time index for regression
    X = np.arange(len(recent_data)).reshape(-1, 1)
    y = recent_data['open'].values
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate forecast
    future_X = np.arange(len(recent_data), len(recent_data) + forecast_days).reshape(-1, 1)
    forecast = model.predict(future_X)
    
    # Add some realistic noise based on historical volatility
    volatility = recent_data['open'].pct_change().std()
    daily_vol = volatility * recent_data['open'].iloc[-1]
    
    # Add cumulative noise that increases with time
    noise = np.random.normal(0, daily_vol * 0.5, forecast_days)
    cumulative_noise = np.cumsum(noise * np.sqrt(np.arange(1, forecast_days + 1) / forecast_days))
    forecast = forecast + cumulative_noise
    
    # Ensure forecast is positive
    forecast = np.maximum(forecast, recent_data['open'].iloc[-1] * 0.5)
    
    current_price = recent_data['open'].iloc[-1]
    forecast = pd.Series(forecast)
    
    # Create future dates
    last_date = clean_data.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    
    # Create plot
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=recent_data.index, 
        y=recent_data['open'],
        mode='lines', 
        name=f'Historical {symbol} Opening Price',
        line=dict(color='blue')
    ))
    
    # Forecasted prices
    fig.add_trace(go.Scatter(
        x=forecast_dates, 
        y=forecast,
        mode='lines', 
        name=f'Trend-Based Forecast {symbol}',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'{symbol} Stock Price Forecast ({forecast_days} Days) - Linear Trend Model',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        showlegend=True
    )
    
    return {
        'symbol': symbol,
        'current_price': current_price,
        'forecast': forecast.tolist(),
        'forecast_dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
        'forecast_min': forecast.min(),
        'forecast_max': forecast.max(),
        'forecast_mean': forecast.mean(),
        'figure': fig,
        'model_summary': 'Linear trend model (SARIMAX fallback)',
        'total_change_percent': ((forecast.iloc[-1] - current_price) / current_price * 100),
        'daily_volatility': volatility * 100,
        'annualized_volatility': volatility * np.sqrt(252) * 100
    }

# Suppress ConvergenceWarning from statsmodels for cleaner output during cross-validation
# Also suppress the specific UserWarning regarding frequency if it reappears
warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning) # Often seen with df slicing

# Assuming calculate_rsi is defined elsewhere, if not, here's a basic implementation:
def calculate_rsi(series, window):
    diff = series.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    ema_up = up.ewm(com=window - 1, adjust=False).mean()
    ema_down = down.ewm(com=window - 1, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def cross_validate_model(symbol, cutoff=2010):
    try:
        # Load and process data
        data = pd.read_json(f"{symbol}.json").T
        data.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        }, inplace=True)

        data.index = pd.to_datetime(data.index)
        data = data.sort_index()

        new_data = data[data.index >= str(cutoff)]
        for col in ['open', 'high', 'low', 'close']:
            new_data[col] = pd.to_numeric(new_data[col])

        # Set frequency to business daily ('B') for stock data, and drop NaNs
        # This is crucial for statsmodels and also for creating future date ranges
        new_data = new_data.asfreq('B').dropna(how='all') # Use how='all' to ensure we don't drop rows with just a few NaNs initially
        new_data = new_data.sort_index()

        # Calculate features - these will introduce NaNs at the beginning
        new_data['log_price'] = np.log(new_data['open'])
        new_data['price_change'] = new_data['open'].pct_change()
        new_data['volatility'] = new_data['price_change'].rolling(window=20).std()
        new_data['rsi'] = calculate_rsi(new_data['open'], window=14)
        new_data['ma_20'] = new_data['open'].rolling(window=20).mean()
        new_data['ma_50'] = new_data['open'].rolling(window=50).mean()
        new_data['price_vs_ma20'] = (new_data['open'] - new_data['ma_20']) / new_data['ma_20']

        # Final dropna after all feature calculations to ensure no NaNs remain
        clean_data = new_data.dropna()

        # Ensure we have enough clean data to proceed
        if len(clean_data) < 200: # Arbitrary minimum for meaningful CV
            raise ValueError(f"Insufficient clean data ({len(clean_data)} points) after preprocessing for {symbol}. Need at least 200.")

        cv_results = []
        exog_features = ['volatility', 'rsi', 'price_vs_ma20']

        print(f"Starting rolling window cross-validation for {symbol}")
        print(f"Training on 5 months, testing on the 6th month, until June 2025 forecast.")

        current_test_start_date = pd.to_datetime(f'{cutoff}-06-01') # Start with test month June 2010
        final_test_month_end_date = pd.to_datetime('2025-06-30')

        fold = 0
        june_2025_full_projection = None # To store the final June 2025 forecast

        while current_test_start_date <= final_test_month_end_date:
            fold += 1
            test_end_date = current_test_start_date + pd.DateOffset(months=1) - pd.DateOffset(days=1)

            # Define the training period: 5 months *before* the current test_start_date
            train_end_date = current_test_start_date - pd.DateOffset(days=1)
            train_start_date = train_end_date - pd.DateOffset(months=5) + pd.DateOffset(days=1)

            train_data = clean_data[
                (clean_data.index >= train_start_date) &
                (clean_data.index <= train_end_date)
            ]

            # Get actual test data that is available within the period
            actual_test_data = clean_data[
                (clean_data.index >= current_test_start_date) &
                (clean_data.index <= test_end_date)
            ]

            # Basic check for sufficient data
            if len(train_data) < 50 or len(actual_test_data) == 0: # Minimum ~2 months of data for training
                print(f"Skipping fold {fold}: Insufficient data in window. Train: {len(train_data)}, Test: {len(actual_test_data)}")
                current_test_start_date += pd.DateOffset(months=1)
                continue

            # Critical: Ensure train_data and its exog features are clean before fitting
            if train_data['log_price'].isnull().any() or train_data[exog_features].isnull().any().any():
                 print(f"Skipping fold {fold}: NaN values found in training data or its exogenous features. Train data length: {len(train_data)}")
                 current_test_start_date += pd.DateOffset(months=1)
                 continue


            print(f"\nFold {fold}")
            print(f"Train: {train_data.index[0].date()} to {train_data.index[-1].date()} ({len(train_data)} points)")
            print(f"Test: {actual_test_data.index[0].date()} to {actual_test_data.index[-1].date()} ({len(actual_test_data)} points, if any actual data)")

            try:
                # Model fitting attempts
                model_configs = [
                    {'order': (1, 1, 1), 'seasonal_order': None},
                    {'order': (2, 1, 2), 'seasonal_order': None},
                    {'order': (1, 0, 1), 'seasonal_order': None}
                ]
                model_fit = None
                for config in model_configs:
                    try:
                        model = SARIMAX(
                            train_data['log_price'],
                            order=config['order'],
                            seasonal_order=config['seasonal_order'],
                            exog=train_data[exog_features],
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        model_fit = model.fit(disp=False, maxiter=200, method='lbfgs')
                        break
                    except Exception as e:
                        # print(f"  Model config {config} failed for fold {fold}: {e}") # Uncomment for verbose debugging
                        continue

                if model_fit is None:
                    print(f"All models failed for fold {fold}, skipping...")
                    current_test_start_date += pd.DateOffset(months=1)
                    continue

                # --- Special Handling for JUNE 2025 Forecast ---
                if current_test_start_date.year == 2025 and current_test_start_date.month == 6:
                    print("\n--- Performing final June 2025 projection ---")
                    # Define the full date range for June 2025 (business days)
                    # Use 'B' (business day frequency) for the forecast dates
                    forecast_full_dates = pd.date_range(start=current_test_start_date, end=test_end_date, freq='B')

                    # Initialize a DataFrame for the complete exogenous variables for the forecast period
                    full_forecast_exog = pd.DataFrame(index=forecast_full_dates, columns=exog_features)

                    # Fill in actual exogenous data for available days in June 2025
                    # Assuming data available till June 4, 2025 (current date from prompt)
                    actual_exog_in_june = clean_data[
                        (clean_data.index >= current_test_start_date) &
                        (clean_data.index <= pd.to_datetime('2025-06-04'))
                    ][exog_features]
                    full_forecast_exog.loc[actual_exog_in_june.index] = actual_exog_in_june

                    # Identify missing dates for which to generate fake data
                    missing_forecast_dates = full_forecast_exog[full_forecast_exog.isnull().any(axis=1)].index

                    if not missing_forecast_dates.empty:
                        print(f"Generating fake exogenous data for {len(missing_forecast_dates)} future dates in June 2025...")
                        train_exog_means = train_data[exog_features].mean()
                        train_exog_stds = train_data[exog_features].std()

                        for date in missing_forecast_dates:
                            for feature in exog_features:
                                # Generate fake data using normal distribution based on training set stats
                                # Add a small floor to std_val to prevent issues with zero or very small std dev
                                std_val = train_exog_stds[feature] if pd.notnull(train_exog_stds[feature]) and train_exog_stds[feature] > 1e-6 else 1e-6 + train_exog_means[feature] * 0.01
                                if std_val < 1e-6: std_val = 1e-6 # Ensure std_val is always positive

                                mean_val = train_exog_means[feature] if pd.notnull(train_exog_means[feature]) else 0.0 # Default mean if NaN

                                full_forecast_exog.loc[date, feature] = np.random.normal(mean_val, std_val)

                    # Ensure the exogenous data for forecasting is sorted by index and has no NaNs
                    test_exog_for_forecast = full_forecast_exog.sort_index().fillna(method='ffill').fillna(method='bfill')
                    if test_exog_for_forecast.isnull().any().any():
                         raise ValueError("NaNs still present in test_exog_for_forecast after generation/fill. Cannot forecast.")

                    # Forecast over the full June 2025 period using the combined (actual + fake) exogenous data
                    forecast_log = model_fit.forecast(steps=len(test_exog_for_forecast), exog=test_exog_for_forecast)
                    forecast_prices_full_june = np.exp(forecast_log)
                    forecast_prices_full_june.index = test_exog_for_forecast.index # Ensure index matches

                    # Store the full June 2025 projection DataFrame
                    june_2025_full_projection = pd.DataFrame({
                        'Projected_Price': forecast_prices_full_june
                    })

                    # For metric calculation, only use actual available data (up to June 4)
                    if not actual_exog_in_june.empty:
                        actual_prices = actual_test_data['open'].values
                        # Match forecast prices to actual_prices dates for metric calculation
                        forecast_prices_for_metrics = forecast_prices_full_june.loc[actual_exog_in_june.index].values

                        mae = mean_absolute_error(actual_prices, forecast_prices_for_metrics)
                        rmse = np.sqrt(mean_squared_error(actual_prices, forecast_prices_for_metrics))
                        mape = np.mean(np.abs((actual_prices - forecast_prices_for_metrics) / actual_prices)) * 100

                        if len(actual_prices) > 1 and len(forecast_prices_for_metrics) > 1:
                            actual_changes = np.diff(actual_prices) > 0
                            forecast_changes = np.diff(forecast_prices_for_metrics) > 0
                            min_len = min(len(actual_changes), len(forecast_changes))
                            directional_accuracy = np.mean(actual_changes[:min_len] == forecast_changes[:min_len]) * 100
                        else:
                            directional_accuracy = np.nan
                    else:
                        mae, rmse, mape, directional_accuracy = np.nan, np.nan, np.nan, np.nan
                        print("No actual data found for June 2025 to calculate metrics.")

                    fold_result = {
                        'fold': fold,
                        'train_start': train_data.index[0],
                        'train_end': train_data.index[-1],
                        'test_start': actual_test_data.index[0] if not actual_test_data.empty else current_test_start_date,
                        'test_end': actual_test_data.index[-1] if not actual_test_data.empty else current_test_start_date, # End of actual data
                        'mae': mae,
                        'rmse': rmse,
                        'mape': mape,
                        'directional_accuracy': directional_accuracy,
                        'actual_prices': actual_prices if 'actual_prices' in locals() else np.array([]),
                        'forecast_prices': forecast_prices_for_metrics if 'forecast_prices_for_metrics' in locals() else np.array([]),
                        'test_dates': actual_test_data.index if not actual_test_data.empty else pd.DatetimeIndex([]),
                        'is_june_2025_forecast': True # Flag for summary
                    }
                    cv_results.append(fold_result)

                    print(f"Metrics for available actual data in June 2025:")
                    print(f"MAE: ${mae:.2f}")
                    print(f"RMSE: ${rmse:.2f}")
                    print(f"MAPE: {mape:.2f}%")
                    print(f"Directional Accuracy: {directional_accuracy:.1f}%")

                else: # Regular folds (not June 2025)
                    test_exog = actual_test_data[exog_features].copy()
                    # Ensure test_exog is clean before forecasting
                    if test_exog.isnull().any().any():
                        print(f"Warning: NaNs found in test_exog for fold {fold}. Attempting to fill...")
                        test_exog = test_exog.fillna(method='ffill').fillna(method='bfill')
                        if test_exog.isnull().any().any(): # If still NaNs after fill
                            raise ValueError(f"NaNs persist in test_exog for fold {fold}. Cannot forecast.")


                    forecast_log = model_fit.forecast(steps=len(actual_test_data), exog=test_exog)
                    forecast_prices = np.exp(forecast_log)
                    forecast_prices.index = actual_test_data.index # Ensure index matches

                    actual_prices = actual_test_data['open'].values
                    mae = mean_absolute_error(actual_prices, forecast_prices)
                    rmse = np.sqrt(mean_squared_error(actual_prices, forecast_prices))
                    mape = np.mean(np.abs((actual_prices - forecast_prices) / actual_prices)) * 100

                    if len(actual_prices) > 1 and len(forecast_prices) > 1:
                        actual_changes = np.diff(actual_prices) > 0
                        forecast_changes = np.diff(forecast_prices) > 0
                        min_len = min(len(actual_changes), len(forecast_changes))
                        directional_accuracy = np.mean(actual_changes[:min_len] == forecast_changes[:min_len]) * 100
                    else:
                        directional_accuracy = np.nan

                    fold_result = {
                        'fold': fold,
                        'train_start': train_data.index[0],
                        'train_end': train_data.index[-1],
                        'test_start': actual_test_data.index[0],
                        'test_end': actual_test_data.index[-1],
                        'mae': mae,
                        'rmse': rmse,
                        'mape': mape,
                        'directional_accuracy': directional_accuracy,
                        'actual_prices': actual_prices,
                        'forecast_prices': forecast_prices,
                        'test_dates': actual_test_data.index
                    }
                    cv_results.append(fold_result)

                    print(f"MAE: ${mae:.2f}")
                    print(f"RMSE: ${rmse:.2f}")
                    print(f"MAPE: {mape:.2f}%")
                    print(f"Directional Accuracy: {directional_accuracy:.1f}%")

            except Exception as e:
                print(f"Error in fold {fold}: {str(e)}")
                pass

            current_test_start_date += pd.DateOffset(months=1)

        if not cv_results:
            raise ValueError("No successful cross-validation folds completed")

        # Exclude the final June 2025 metrics from overall averages if you only want past performance
        avg_metrics_folds = [r for r in cv_results if 'is_june_2025_forecast' not in r or r['is_june_2025_forecast'] == False]
        if not avg_metrics_folds: # If only June 2025 fold was done, use it despite partial actual data
            print("Warning: Only the June 2025 fold (with partial actual data) was completed for metrics. Averages might not be representative.")
            avg_metrics_folds = cv_results

        avg_metrics = {
            'avg_mae': np.nanmean([r['mae'] for r in avg_metrics_folds]),
            'avg_rmse': np.nanmean([r['rmse'] for r in avg_metrics_folds]),
            'avg_mape': np.nanmean([r['mape'] for r in avg_metrics_folds]),
            'avg_directional_accuracy': np.nanmean([r['directional_accuracy'] for r in avg_metrics_folds if not np.isnan(r['directional_accuracy'])]),
            'std_mae': np.nanstd([r['mae'] for r in avg_metrics_folds]),
            'std_rmse': np.nanstd([r['rmse'] for r in avg_metrics_folds]),
            'std_mape': np.nanstd([r['mape'] for r in avg_metrics_folds]),
            'n_folds': len(avg_metrics_folds)
        }

        return {
            'symbol': symbol,
            'cv_results': cv_results,
            'avg_metrics': avg_metrics,
            'june_2025_projection': june_2025_full_projection
        }

    except Exception as e:
        raise Exception(f"Error in cross-validation for {symbol}: {str(e)}")

def display_cv_summary(cv_results):
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION SUMMARY FOR {cv_results['symbol']}")
    print(f"{'='*60}")
    print(f"Number of historical folds completed for average metrics: {cv_results['avg_metrics']['n_folds']}")
    print(f"Forecast horizon (per fold): 1 month")
    print(f"\nAverage Historical Performance Metrics:")
    print(f"   Mean Absolute Error (MAE): ${cv_results['avg_metrics']['avg_mae']:.2f} \u00B1 ${cv_results['avg_metrics']['std_mae']:.2f}")
    print(f"   Root Mean Square Error (RMSE): ${cv_results['avg_metrics']['avg_rmse']:.2f} \u00B1 ${cv_results['avg_metrics']['std_rmse']:.2f}")
    print(f"   Mean Absolute Percentage Error (MAPE): {cv_results['avg_metrics']['avg_mape']:.2f}% \u00B1 {cv_results['avg_metrics']['std_mape']:.2f}%")
    print(f"   Directional Accuracy: {cv_results['avg_metrics']['avg_directional_accuracy']:.1f}%")

    # Performance interpretation based on historical MAPE
    mape = cv_results['avg_metrics']['avg_mape']
    if mape < 5:
        performance = "Excellent"
    elif mape < 10:
        performance = "Good"
    elif mape < 20:
        performance = "Reasonable"
    else:
        performance = "Poor"

    print(f"\nHistorical Model Performance Rating: {performance}")
    print(f"Note: MAPE < 10% is generally considered good for stock price forecasting")

def display_june_2025_projection(cv_results):
    if 'june_2025_projection' in cv_results and cv_results['june_2025_projection'] is not None:
        print(f"\n{'='*60}")
        print(f"JUNE 2025 PROJECTED STOCK PRICES FOR {cv_results['symbol']}")
        print(f"{'='*60}")
        print(cv_results['june_2025_projection'].to_string())
        print(f"{'='*60}")
    else:
        print("\nNo June 2025 projection available in the results.")

    
def display_forecast_results(results):
    print(f"\n{'='*50}")
    print(f"FORECAST RESULTS FOR {results['symbol']}")
    print(f"{'='*50}")
    print(f"Current Price: ${results['current_price']:.2f}")
    print(f"Forecast Range: ${results['forecast_min']:.2f} - ${results['forecast_max']:.2f}")
    print(f"Average Forecasted Price: ${results['forecast_mean']:.2f}")
    print(f"Total Price Change: {results['total_change_percent']:+.2f}%")
    print(f"Daily Volatility: {results['daily_volatility']:.2f}%")
    print(f"Annualized Volatility: {results['annualized_volatility']:.2f}%")

    # Show the plot
    results['figure'].show()

# Example usage
if __name__ == "__main__":
    # Example with user input
    symbol = input("Enter stock symbol (e.g., MSFT, GOOG): ").upper()
    
    print(f"Starting cross-validation for {symbol}...")
    try:
        results = cross_validate_model(symbol)
        display_cv_summary(results)
        display_june_2025_projection(results)
    except Exception as e:
        print(f"An error occurred: {e}")
