
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
import json
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
                # Generate constrained forecast
        try:
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
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def simple_trend_forecast(clean_data, symbol, forecast_days):
    """
    Simple trend-based forecast as fallback when SARIMAX fails
    """
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

def cross_validate_model(symbol, cutoff=2016, n_splits=5, forecast_horizon=30):
    """
    Perform time series cross-validation on the forecast model
    
    Args:
        symbol (str): Stock symbol
        cutoff (int): Year cutoff for data filtering
        n_splits (int): Number of cross-validation splits
        forecast_horizon (int): Number of days to forecast in each split
    
    Returns:
        dict: Cross-validation results including metrics and fold details
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    try:
        # Load and process data (same preprocessing as main function)
        data = pd.read_json(f"{symbol}.json").T
        data.rename(columns={
            '1. open': 'open', 
            '2. high': 'high', 
            '3. low': 'low', 
            '4. close': 'close', 
            '5. volume': 'volume'
        }, inplace=True)

        new_data = data[data.index > str(cutoff)]
        for col in ['open', 'high', 'low', 'close']:
            new_data[col] = pd.to_numeric(new_data[col])

        new_data.index = pd.to_datetime(new_data.index)
        new_data = new_data.sort_index()

        # Calculate features
        new_data['log_price'] = np.log(new_data['open'])
        new_data['price_change'] = new_data['open'].pct_change()
        new_data['volatility'] = new_data['price_change'].rolling(window=20).std()
        new_data['rsi'] = calculate_rsi(new_data['open'], window=14)
        new_data['ma_20'] = new_data['open'].rolling(window=20).mean()
        new_data['ma_50'] = new_data['open'].rolling(window=50).mean()
        new_data['price_vs_ma20'] = (new_data['open'] - new_data['ma_20']) / new_data['ma_20']
        
        clean_data = new_data.dropna()
        
        if len(clean_data) < 200:
            raise ValueError(f"Insufficient data for cross-validation. Need at least 200 data points.")

        # Time series cross-validation setup
        total_points = len(clean_data)
        min_train_size = total_points // 2  # Minimum 50% for training
        test_size = forecast_horizon
        
        cv_results = []
        exog_features = ['volatility', 'rsi', 'price_vs_ma20']
        
        print(f"Starting {n_splits}-fold time series cross-validation for {symbol}")
        print(f"Forecast horizon: {forecast_horizon} days")
        
        for fold in range(n_splits):
            # Calculate split points
            split_point = min_train_size + fold * (total_points - min_train_size - test_size) // (n_splits - 1)
            train_end = split_point
            test_start = split_point
            test_end = min(split_point + test_size, total_points)
            
            if test_end - test_start < forecast_horizon:
                continue
                
            # Split data
            train_data = clean_data.iloc[:train_end]
            test_data = clean_data.iloc[test_start:test_end]
            
            print(f"\nFold {fold + 1}/{n_splits}")
            print(f"Train: {train_data.index[0].date()} to {train_data.index[-1].date()} ({len(train_data)} points)")
            print(f"Test: {test_data.index[0].date()} to {test_data.index[-1].date()} ({len(test_data)} points)")
            
            try:
                # Try to fit model on training data with fallback
                model_configs = [
                    {'order': (1, 1, 1), 'seasonal_order': None},
                    {'order': (2, 1, 2), 'seasonal_order': None},
                    {'order': (1, 0, 1), 'seasonal_order': None}
                ]
                
                model_fit = None
                for config in model_configs:
                    try:
                        if config['seasonal_order'] is None:
                            model = SARIMAX(
                                train_data['log_price'],
                                order=config['order'],
                                exog=train_data[exog_features],
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                        else:
                            model = SARIMAX(
                                train_data['log_price'],
                                order=config['order'],
                                seasonal_order=config['seasonal_order'],
                                exog=train_data[exog_features],
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                        
                        model_fit = model.fit(disp=False, maxiter=100, method='lbfgs')
                        break
                    except:
                        continue
                
                if model_fit is None:
                    print(f"All models failed for fold {fold + 1}, skipping...")
                    continue
                
                # Generate exogenous variables for test period
                test_exog = test_data[exog_features].copy()
                
                # Forecast
                forecast_log = model_fit.forecast(steps=len(test_data), exog=test_exog)
                forecast_prices = np.exp(forecast_log)
                
                # Calculate metrics
                actual_prices = test_data['open'].values
                mae = mean_absolute_error(actual_prices, forecast_prices)
                rmse = np.sqrt(mean_squared_error(actual_prices, forecast_prices))
                mape = np.mean(np.abs((actual_prices - forecast_prices) / actual_prices)) * 100
                
                # Calculate directional accuracy
                actual_changes = np.diff(actual_prices) > 0
                forecast_changes = np.diff(forecast_prices) > 0
                directional_accuracy = np.mean(actual_changes == forecast_changes) * 100
                
                fold_result = {
                    'fold': fold + 1,
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'directional_accuracy': directional_accuracy,
                    'actual_prices': actual_prices,
                    'forecast_prices': forecast_prices,
                    'test_dates': test_data.index
                }
                
                cv_results.append(fold_result)
                
                print(f"MAE: ${mae:.2f}")
                print(f"RMSE: ${rmse:.2f}")
                print(f"MAPE: {mape:.2f}%")
                print(f"Directional Accuracy: {directional_accuracy:.1f}%")
                
            except Exception as e:
                print(f"Error in fold {fold + 1}: {str(e)}")
                continue
        
        if not cv_results:
            raise ValueError("No successful cross-validation folds completed")
        
        # Calculate average metrics
        avg_metrics = {
            'avg_mae': np.mean([r['mae'] for r in cv_results]),
            'avg_rmse': np.mean([r['rmse'] for r in cv_results]),
            'avg_mape': np.mean([r['mape'] for r in cv_results]),
            'avg_directional_accuracy': np.mean([r['directional_accuracy'] for r in cv_results]),
            'std_mae': np.std([r['mae'] for r in cv_results]),
            'std_rmse': np.std([r['rmse'] for r in cv_results]),
            'std_mape': np.std([r['mape'] for r in cv_results]),
            'n_folds': len(cv_results)
        }
        
        return {
            'symbol': symbol,
            'cv_results': cv_results,
            'avg_metrics': avg_metrics,
            'forecast_horizon': forecast_horizon
        }
        
    except Exception as e:
        raise Exception(f"Error in cross-validation for {symbol}: {str(e)}")

def plot_cv_results(cv_results):
    """
    Plot cross-validation results showing actual vs predicted for each fold
    
    Args:
        cv_results (dict): Results from cross_validate_model function
    """
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, fold_result in enumerate(cv_results['cv_results']):
        color = colors[i % len(colors)]
        
        # Plot actual prices
        fig.add_trace(go.Scatter(
            x=fold_result['test_dates'],
            y=fold_result['actual_prices'],
            mode='lines',
            name=f'Actual - Fold {fold_result["fold"]}',
            line=dict(color=color, width=2)
        ))
        
        # Plot forecasted prices
        fig.add_trace(go.Scatter(
            x=fold_result['test_dates'],
            y=fold_result['forecast_prices'],
            mode='lines',
            name=f'Forecast - Fold {fold_result["fold"]}',
            line=dict(color=color, width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=f'Cross-Validation Results - {cv_results["symbol"]} ({cv_results["forecast_horizon"]} day forecasts)',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        showlegend=True,
        height=600
    )
    
    fig.show()

def display_cv_summary(cv_results):
    """
    Display cross-validation summary statistics
    
    Args:
        cv_results (dict): Results from cross_validate_model function
    """
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION SUMMARY FOR {cv_results['symbol']}")
    print(f"{'='*60}")
    print(f"Number of folds completed: {cv_results['avg_metrics']['n_folds']}")
    print(f"Forecast horizon: {cv_results['forecast_horizon']} days")
    print(f"\nAverage Performance Metrics:")
    print(f"  Mean Absolute Error (MAE): ${cv_results['avg_metrics']['avg_mae']:.2f} ± ${cv_results['avg_metrics']['std_mae']:.2f}")
    print(f"  Root Mean Square Error (RMSE): ${cv_results['avg_metrics']['avg_rmse']:.2f} ± ${cv_results['avg_metrics']['std_rmse']:.2f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {cv_results['avg_metrics']['avg_mape']:.2f}% ± {cv_results['avg_metrics']['std_mape']:.2f}%")
    print(f"  Directional Accuracy: {cv_results['avg_metrics']['avg_directional_accuracy']:.1f}%")
    
    # Performance interpretation
    mape = cv_results['avg_metrics']['avg_mape']
    if mape < 5:
        performance = "Excellent"
    elif mape < 10:
        performance = "Good"
    elif mape < 20:
        performance = "Reasonable"
    else:
        performance = "Poor"
    
    print(f"\nModel Performance Rating: {performance}")
    print(f"Note: MAPE < 10% is generally considered good for stock price forecasting")

def display_forecast_results(results):
    """
    Display forecast results in a formatted way

    Args:
        results (dict): Results from forecast_stock_price function
    """
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
    
    # Ask user if they want to run cross-validation
    run_cv = input("Run cross-validation first? (y/n): ").lower().strip()

    try:
        # Run cross-validation if requested
        if run_cv == 'y':
            print("Running cross-validation backtesting...")
            cv_results = cross_validate_model(symbol, cutoff=2016, n_splits=5, forecast_horizon=30)
            display_cv_summary(cv_results)
            plot_cv_results(cv_results)
            
            # Ask if user wants to continue with forecast
            continue_forecast = input("\nContinue with forward forecast? (y/n): ").lower().strip()
            if continue_forecast != 'y':
                exit()

        # Run main forecast
        print("Generating forward forecast...")
        results = forecast_stock_price(symbol, cutoff=2016, forecast_days=30)
        display_forecast_results(results)

        # Optionally save results to CSV
        forecast_df = pd.DataFrame({
            'Date': results['forecast_dates'],
            'Forecasted_Price': results['forecast']
        })
        forecast_df.to_csv(f'{symbol}_forecast.csv', index=False)
        print(f"\nForecast saved to {symbol}_forecast.csv")

    except Exception as e:
        print(f"Error: {e}")
