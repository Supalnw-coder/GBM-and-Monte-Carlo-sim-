import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tvDatafeed import TvDatafeed, Interval
from scipy.stats import skew, kurtosis

# -------------------------------------------------------------------
# ส่วนดึงข้อมูลและตั้งค่า
# -------------------------------------------------------------------
tv = TvDatafeed()

symbol = input("ใส่รหัสสินทรัพย์ (e.g., AAPL): ") or "AAPL"
exchange = input("ใส่ตลาด (e.g., NASDAQ): ") or "NASDAQ"

try:
    df_price = tv.get_hist(symbol=symbol, exchange=exchange,
                           interval=Interval.in_daily, n_bars=1000000)
    if df_price is None:
        raise ValueError("ไม่พบข้อมูลจาก TradingView")
except Exception as e:
    print(f"ดึงข้อมูลไม่สำเร็จ: {e} ใช้ค่าเริ่มต้น AAPL/NASDAQ")
    df_price = tv.get_hist(symbol="AAPL", exchange="NASDAQ",
                           interval=Interval.in_daily, n_bars=1000000)

df_price = df_price[['close']]

START_DATE = input("ใส่วันที่เริ่มต้น (เช่น YYYY-MM-DD, เว้นว่าง=ไกลที่สุดที่จะไกลได้): ") or None
END_DATE = input("ใส่วันที่สิ้นสุด (เว้นว่าง=ถึงปัจจุบัน): ") or None

df_price = df_price.loc[START_DATE:END_DATE]
print(f"ข้อมูลราคาปิดของ {symbol} ({exchange}) ตั้งแต่ {df_price.index[0].date()} ถึง {df_price.index[-1].date()}")

# -------------------------------------------------------------------
# คำนวณค่าพื้นฐาน
# -------------------------------------------------------------------
df_price['Daily Return'] = np.log(df_price['close'] / df_price['close'].shift(1))
df_daily_return = df_price.dropna()

mean_return = df_daily_return['Daily Return'].mean()
volatility = df_daily_return['Daily Return'].std()
last_price = df_daily_return['close'].iloc[-1]

num_simulations = int(input("ใส่จำนวนการจำลอง (เว้นว่าง=1000): ") or 1000)
num_days = int(input("ใส่จำนวนวันสำหรับการจำลอง (เว้นว่าง=252): ") or 252)
market_days_per_year = int(input("ใส่จำนวนวันซื้อขายต่อปีของตลาด (เว้นว่าง=252): ") or 252)
dT = 1 / market_days_per_year

# -------------------------------------------------------------------
# Monte Carlo Simulation (GBM)
# -------------------------------------------------------------------
rand_norm = np.random.normal(0, 1, size=(num_days, num_simulations))
price_paths = last_price * np.exp(np.cumsum((mean_return - 0.5 * volatility**2) * dT
                                            + volatility * np.sqrt(dT) * rand_norm, axis=0))
price_paths = np.vstack([np.full((1, num_simulations), last_price), price_paths])
simulation_df = pd.DataFrame(price_paths, columns=[f"Sim{i+1}" for i in range(num_simulations)])

# -------------------------------------------------------------------
# คำนวณเพิ่มเติม (Outlier, Percentile, Skewness, Kurtosis, Drawdown)
# -------------------------------------------------------------------
def calc_outlier_pct(df, mean, std, k):
    return (df[(df['Daily Return'] >= mean + k*std) | (df['Daily Return'] <= mean - k*std)].shape[0] / df.shape[0]) * 100

out_1sd = calc_outlier_pct(df_daily_return, mean_return, volatility, 1)
out_2sd = calc_outlier_pct(df_daily_return, mean_return, volatility, 2)
out_3sd = calc_outlier_pct(df_daily_return, mean_return, volatility, 3)

percentile_5 = simulation_df.quantile(0.05, axis=1)
percentile_95 = simulation_df.quantile(0.95, axis=1)

# --- Skewness & Kurtosis ---
skew_val = skew(df_daily_return['Daily Return'])
kurt_val = kurtosis(df_daily_return['Daily Return'], fisher=True)

# --- Max Drawdown ---
rolling_max = df_price['close'].cummax()
drawdown = (df_price['close'] - rolling_max) / rolling_max
max_drawdown = drawdown.min()

end_date = drawdown.idxmin()
start_date = (df_price['close'][:end_date]).idxmax()
recovery_date = df_price[df_price.index > end_date]['close'][df_price['close'] >= df_price.loc[start_date, 'close']].first_valid_index()
recovery_period = (recovery_date - end_date).days if recovery_date is not None else None

# -------------------------------------------------------------------
# คำนวณ VaR (Value at Risk)
# -------------------------------------------------------------------
final_simulated_prices = simulation_df.iloc[-1]
simulated_return_am = (final_simulated_prices / last_price) - 1

try:
    user_confidence = float(input("ป้อนระดับความเชื่อมั่นสำหรับ VaR (เช่น 95 หรือ 99): ").strip())
    if not (0 < user_confidence < 100):
        raise ValueError
except ValueError:
    print("❗ ค่าที่ป้อนไม่ถูกต้อง ใช้ค่าเริ่มต้น 95%")
    user_confidence = 95.0

cl = user_confidence / 100
percentile = (1 - cl) * 100
VaR_value = np.percentile(simulated_return_am, percentile)

# -------------------------------------------------------------------
# แสดงผลทั้งหมดในหน้าเดียว (3 กราฟ + ตาราง)
# -------------------------------------------------------------------
fig = plt.figure(constrained_layout=True, figsize=(17, 10))
gs = GridSpec(2, 2, figure=fig)

#  กราฟ 1: Histogram of Daily Returns
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(df_daily_return['Daily Return'] * 100, bins=30, color='skyblue', edgecolor='black', ax=ax1)
ax1.axvline(mean_return * 100, color='red', linestyle='-', label=f'Mean: {mean_return*100:.2f}%')
ax1.axvline(mean_return*100 + 2*volatility*100, color='blue', linestyle='--', label='+2 SD')
ax1.axvline(mean_return*100 - 2*volatility*100, color='blue', linestyle='--', label='-2 SD')
ax1.set_title(f'Histogram of Daily Return ({symbol})', fontsize=12)
ax1.set_xlabel('Daily Return (%)')
ax1.legend()

#  กราฟ 2: Simulated Price Paths
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(simulation_df, color='red', alpha=0.05)
ax2.plot(percentile_5, color='green', linestyle='--', label='5th Percentile')
ax2.plot(percentile_95, color='orange', linestyle='--', label='95th Percentile')
ax2.set_title(f'{symbol} Monte Carlo Simulation ({num_simulations} paths, {num_days} days)')
ax2.set_xlabel('Days')
ax2.set_ylabel('Price (USD)')
ax2.legend()

#  กราฟ 3: VaR Histogram 
ax3 = fig.add_subplot(gs[1, 0])
sns.histplot(simulated_return_am * 100, bins=50, color='lightblue', edgecolor='black', ax=ax3)

if VaR_value < 0:
    ax3.axvline(VaR_value * 100, color='red', linestyle='--', linewidth=2,
                label=f'VaR {user_confidence:.0f}%: {VaR_value*100:.2f}%')
elif VaR_value == 0:
    ax3.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Break-even (0%)')
else:
    ax3.axvline(VaR_value * 100, color='green', linestyle='--', linewidth=2,
                label=f'Min Return @ {user_confidence:.0f}%: +{VaR_value*100:.2f}%')

ax3.set_title(f'Value at Risk Analysis ({symbol})', fontsize=12)
ax3.set_xlabel('Simulated Arithmetic Return (%)')
ax3.legend()

#  ตาราง Summary
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')
summary_table = [
    ['Mean (μ)', f'{mean_return*100:.4f}%'],
    ['Volatility (σ)', f'{volatility*100:.4f}%'],
    ['Skewness', f'{skew_val:.4f}'],
    ['Kurtosis', f'{kurt_val:.4f}'],
    ['Outside 1 SD', f'{out_1sd:.2f}%'],
    ['Outside 2 SD', f'{out_2sd:.2f}%'],
    ['Outside 3 SD', f'{out_3sd:.2f}%'],
    ['Max Drawdown', f'{max_drawdown*100:.2f}%'],
    ['Drawdown Period', f'{start_date.date()} to {end_date.date()}'],
    ['Recovery Time', f'{recovery_period if recovery_period is not None else "Not recovered"} Day'],
    ['5th Percentile Return', f'{(percentile_5.iloc[-1]/last_price - 1)*100:.2f}%'],
    ['95th Percentile Return', f'{(percentile_95.iloc[-1]/last_price - 1)*100:.2f}%'],
    [f'VaR {user_confidence:.0f}%', f'{VaR_value*100:.2f}%']  # ใช้ค่าตรงกับกราฟแน่นอน
]
table = ax4.table(cellText=summary_table, colLabels=['Metric', 'Value'],
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1.3, 1.5)

plt.suptitle(f'Monte Carlo Simulation Dashboard: {symbol}', fontsize=14)
plt.show()

# -------------------------------------------------------------------
# สรุปผลในคอนโซล
# -------------------------------------------------------------------
print("\n====================== Value at Risk (VaR) ===========================")
if VaR_value < 0:
    print(f"สรุปได้ว่าในระยะเวลา {num_days} วันทำการ มีโอกาส {100 - user_confidence:.0f}% ที่ {symbol} ")
    print(f"จะขาดทุนมากกว่า {abs(VaR_value*100):.2f}% จากมูลค่าปัจจุบัน ({last_price:.2f} USD)")
elif VaR_value == 0:
    print("VaR = 0% → ไม่มีขาดทุนใน quantile ล่างสุด (Break-even)")
else:
    print(f"VaR = +{VaR_value*100:.2f}% → ผลตอบแทนต่ำสุดยังเป็นบวก ไม่มีความเสี่ยงขาดทุน")
print("=======================================================================")

