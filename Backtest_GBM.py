import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tvDatafeed import TvDatafeed, Interval
from datetime import datetime
 
# Set matplotlib to use default font (no Thai characters)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ตั้งค่าพารามิเตอร์
STOCK = "VOO"
EXCHANGE = "AMEX"
FORECAST_YEARS = 1  # จำนวนปีที่ต้องการทำนาย (เปลี่ยนเป็น 5 ได้)
START_YEAR = 1990
END_YEAR = 2024
NUM_SIMULATIONS = 10000
TRADING_DAYS_PER_YEAR = 252

# Fetch data from TradingView
print("Fetching data from TradingView...")
tv = TvDatafeed()
df_his_price = tv.get_hist(symbol=STOCK, exchange=EXCHANGE, interval=Interval.in_daily, n_bars=10000000)
df_his_price = df_his_price[['open', 'close']]
df_his_price.index = pd.to_datetime(df_his_price.index)

# Calculate Daily Return
df_his_price['Daily Return'] = np.log(df_his_price['close'] / df_his_price['close'].shift(1))
df_his_price = df_his_price.dropna()

print(f"Data from {df_his_price.index.min().date()} to {df_his_price.index.max().date()}")

# Function to simulate price with GBM
def simulate_gbm(last_price, mean, sd, num_days, num_simulations, actual_trading_days):
    """
    Simulate price using Geometric Brownian Motion
    actual_trading_days: actual number of trading days in the training period
    """
    deltaT = 1 / actual_trading_days  # Use actual trading days instead of fixed 252
    rand_norm = np.random.normal(0, 1, size=(num_days, num_simulations))
    
    price_paths = last_price * np.exp(
        np.cumsum((mean - 0.5 * sd**2) * deltaT + sd * np.sqrt(deltaT) * rand_norm, axis=0)
    )
    
    # Add initial price
    price_paths = np.vstack([np.full((1, num_simulations), last_price), price_paths])
    
    return price_paths

# Rolling Window Backtest function
def rolling_window_backtest(df, start_year, end_year, forecast_years):
    """
    Perform Rolling Window Backtest
    """
    results = []
    
    current_year = start_year
    
    while current_year + forecast_years <= end_year:
        # Define training period - up to end of (current_year + forecast_years - 1)
        train_end_date = f"{current_year + forecast_years - 1}-12-31"
        train_data = df[df.index <= train_end_date]
        
        if len(train_data) < 30:  # Need at least 30 days of data
            current_year += forecast_years
            continue
        
        # Define forecast period - from start of (current_year + forecast_years) to end of that period
        forecast_start_date = f"{current_year + forecast_years}-01-01"
        forecast_end_date = f"{current_year + 2 * forecast_years - 1}-12-31"
        
        # Get actual forecast period data to count trading days
        forecast_data = df[(df.index >= forecast_start_date) & (df.index <= forecast_end_date)]
        
        # Use actual number of trading days in forecast period
        num_forecast_days = len(forecast_data) if len(forecast_data) > 0 else forecast_years * TRADING_DAYS_PER_YEAR
        
        # Calculate actual trading days per year in training data
        train_years = (train_data.index[-1] - train_data.index[0]).days / 365.25
        actual_trading_days_per_year = len(train_data) / train_years if train_years > 0 else TRADING_DAYS_PER_YEAR
        
        # Calculate parameters from training data
        mean = train_data['Daily Return'].mean()
        sd = train_data['Daily Return'].std()
        last_price = train_data['close'].iloc[-1]
        
        # Simulate prices using actual trading days
        price_paths = simulate_gbm(last_price, mean, sd, num_forecast_days, NUM_SIMULATIONS, actual_trading_days_per_year)
        
        # Calculate percentiles
        final_prices = price_paths[-1, :]  # Final price of each simulation
        percentile_5 = np.percentile(final_prices, 5)
        percentile_95 = np.percentile(final_prices, 95)
        percentile_50 = np.percentile(final_prices, 50)
        
        # Find actual price in forecast period
        actual_data = forecast_data
        
        if len(actual_data) > 0:
            actual_price = actual_data['close'].iloc[-1]
            actual_start_date = actual_data.index[0]
            actual_end_date = actual_data.index[-1]
        else:
            actual_price = None
            actual_start_date = pd.Timestamp(forecast_start_date)
            actual_end_date = pd.Timestamp(forecast_end_date)
        
        results.append({
            'year': current_year,
            'forecast_period': f"{current_year + forecast_years}",
            'last_train_price': last_price,
            'predicted_5th': percentile_5,
            'predicted_50th': percentile_50,
            'predicted_95th': percentile_95,
            'actual_price': actual_price,
            'mean': mean,
            'sd': sd,
            'train_end_date': train_data.index[-1],
            'forecast_start_date': actual_start_date,
            'forecast_end_date': actual_end_date,
            'price_paths': price_paths,
            'actual_trading_days': actual_trading_days_per_year,
            'num_forecast_days': num_forecast_days
        })
        
        print(f"Year {current_year}: Train until {train_data.index[-1].date()}, forecast {actual_start_date.date()} to {actual_end_date.date()}")
        print(f"  Last price: ${last_price:.2f}")
        print(f"  Trading days/year: {actual_trading_days_per_year:.1f}")
        print(f"  Forecast days: {num_forecast_days}")
        print(f"  Forecast 5th-95th percentile: ${percentile_5:.2f} - ${percentile_95:.2f}")
        if actual_price:
            print(f"  Actual price: ${actual_price:.2f}")
        print()
        
        current_year += forecast_years
    
    return pd.DataFrame(results)

# Run Backtest
print(f"\n{'='*60}")
print(f"Starting Backtest: Forecasting every {FORECAST_YEARS} year(s)")
print(f"{'='*60}\n")

results_df = rolling_window_backtest(df_his_price, START_YEAR, END_YEAR, FORECAST_YEARS)

# Display results in chart
fig, ax = plt.subplots(figsize=(16, 8))

# Clear any cached fonts
plt.rcParams.update({'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica']})

# Plot actual price
years = df_his_price.index.year
ax.plot(df_his_price.index, df_his_price['close'], 
        color='black', linewidth=2, label='Actual Price', zorder=5)

# Plot forecast range for each year
colors = plt.cm.rainbow(np.linspace(0, 1, len(results_df)))

for idx, row in results_df.iterrows():
    year = row['year']
    forecast_year = int(row['forecast_period'])
    
    # Use actual forecast dates from results
    forecast_start_date = row['forecast_start_date']
    forecast_end_date = row['forecast_end_date']
    
    # Get actual forecast data for this specific period only
    actual_forecast_data = df_his_price[(df_his_price.index >= forecast_start_date) & 
                                         (df_his_price.index <= forecast_end_date)]
    
    if len(actual_forecast_data) > 0:
        # Use actual trading days as x-axis
        date_range = actual_forecast_data.index
        
        # price_paths includes starting price at [0], so we use [1:] for forecast only
        price_paths_forecast = row['price_paths'][1:]
        
        # Match lengths
        min_len = min(len(date_range), len(price_paths_forecast))
        date_range = date_range[:min_len]
        price_paths_forecast = price_paths_forecast[:min_len]
        
        # Calculate percentiles
        percentile_5_path = np.percentile(price_paths_forecast, 5, axis=1)
        percentile_95_path = np.percentile(price_paths_forecast, 95, axis=1)
        
        # Plot forecast range
        ax.fill_between(date_range, percentile_5_path, percentile_95_path,
                         alpha=0.2, color=colors[idx], 
                         label=f"Forecast {year}->{forecast_year}")
        
        # Plot 5th and 95th percentile lines
        ax.plot(date_range, percentile_5_path, '-', color=colors[idx], linewidth=1, alpha=0.6, linestyle=':')
        ax.plot(date_range, percentile_95_path, '-', color=colors[idx], linewidth=1, alpha=0.6, linestyle=':')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Price (USD)', fontsize=12)
ax.set_title(f'VaR Backtest: {STOCK} Price Forecast with Rolling Window ({FORECAST_YEARS} Year)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary
print(f"\n{'='*60}")
print("Backtest Summary")
print(f"{'='*60}\n")

results_summary = results_df[['year', 'forecast_period', 'last_train_price', 
                               'predicted_5th', 'predicted_50th', 'predicted_95th', 
                               'actual_price']].copy()

# Calculate whether actual price is in range
results_summary['in_range'] = results_summary.apply(
    lambda x: (x['predicted_5th'] <= x['actual_price'] <= x['predicted_95th']) 
    if pd.notna(x['actual_price']) else None, axis=1
)

print(results_summary.to_string(index=False))

# Calculate Coverage (% of actual price within 5-95 percentile range)
valid_predictions = results_summary['in_range'].dropna()
if len(valid_predictions) > 0:
    coverage = (valid_predictions.sum() / len(valid_predictions)) * 100
    print(f"{'='*60}")
    print(f"Coverage: {coverage:.2f}% ({int(valid_predictions.sum())}/{len(valid_predictions)} times)")
    print(f"Should be close to 90% for 5th-95th percentile")
    print(f"{'='*60}")