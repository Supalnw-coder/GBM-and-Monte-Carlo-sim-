import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from tvDatafeed import TvDatafeed, Interval

# ---------------------------------------------------------------
# ตั้งค่าหน้าเว็บ
# ---------------------------------------------------------------
st.set_page_config(page_title="Monte Carlo Simulation Dashboard", layout="wide")
st.title("📊 Monte Carlo Simulation Dashboard ")

# ---------------------------------------------------------------
# ส่วนรับ Input (แทน input() เดิม)
# ---------------------------------------------------------------
st.sidebar.header("⚙️ Simulation Parameters")

symbol = st.sidebar.text_input("ใส่รหัสสินทรัพย์ (e.g., AAPL):", "AAPL")
exchange = st.sidebar.text_input("ใส่ตลาด (e.g., NASDAQ):", "NASDAQ")
START_DATE = st.sidebar.text_input("ใส่วันที่เริ่มต้น (YYYY-MM-DD):", "")
END_DATE = st.sidebar.text_input("ใส่วันที่สิ้นสุด (YYYY-MM-DD):", "")
num_simulations = st.sidebar.number_input("จำนวนการจำลอง", min_value=100, max_value=10000, value=1000, step=100)
num_days = st.sidebar.number_input("จำนวนวันสำหรับการจำลอง", min_value=30, max_value=1000, value=252, step=10)
market_days_per_year = st.sidebar.number_input("จำนวนวันซื้อขายต่อปี", min_value=200, max_value=365, value=252, step=1)
user_confidence = st.sidebar.slider("ระดับความเชื่อมั่นสำหรับ VaR (%)", 90, 99, 95)

# ---------------------------------------------------------------
# ดึงข้อมูลจาก TradingView 
# ---------------------------------------------------------------
tv = TvDatafeed()
try:
    df_price = tv.get_hist(symbol=symbol, exchange=exchange,
                           interval=Interval.in_daily, n_bars=1000000)
    if df_price is None:
        raise ValueError("ไม่พบข้อมูลจาก TradingView")
except Exception as e:
    st.warning(f"⚠️ ดึงข้อมูลไม่สำเร็จ: {e}\nใช้ค่าเริ่มต้น AAPL/NASDAQ")
    df_price = tv.get_hist(symbol="AAPL", exchange="NASDAQ",
                           interval=Interval.in_daily, n_bars=1000000)

df_price = df_price[['close']]
df_price = df_price.loc[START_DATE or None:END_DATE or None]
st.write(f"**ข้อมูลราคาของปิด {symbol} จาก {exchange} ตั้งแต่ {df_price.index[0].date()} ถึง {df_price.index[-1].date()}**")

# ---------------------------------------------------------------
# คำนวณผลตอบแทนรายวัน 
# ---------------------------------------------------------------
df_price['Daily Return'] = np.log(df_price['close'] / df_price['close'].shift(1))
df_daily_return = df_price.dropna()

mean_return = df_daily_return['Daily Return'].mean()
volatility = df_daily_return['Daily Return'].std()
last_price = df_daily_return['close'].iloc[-1]
dT = 1 / market_days_per_year

# ---------------------------------------------------------------
# Monte Carlo Simulation (GBM)
# ---------------------------------------------------------------
rand_norm = np.random.normal(0, 1, size=(num_days, num_simulations))
price_paths = last_price * np.exp(
    np.cumsum((mean_return - 0.5 * volatility**2) * dT
              + volatility * np.sqrt(dT) * rand_norm, axis=0)
)
price_paths = np.vstack([np.full((1, num_simulations), last_price), price_paths])
simulation_df = pd.DataFrame(price_paths, columns=[f"Sim{i+1}" for i in range(num_simulations)])

# ---------------------------------------------------------------
# การคำนวณ Outlier + Percentile (เหมือนเดิม)
# ---------------------------------------------------------------
def calc_outlier_pct(df, mean, std_dev, k):
    return (df[(df['Daily Return'] >= mean + k*std_dev) |
               (df['Daily Return'] <= mean - k*std_dev)].shape[0] / df.shape[0]) * 100

out_1sd = calc_outlier_pct(df_daily_return, mean_return, volatility, 1)
out_2sd = calc_outlier_pct(df_daily_return, mean_return, volatility, 2)
out_3sd = calc_outlier_pct(df_daily_return, mean_return, volatility, 3)

percentile_5 = simulation_df.quantile(0.05, axis=1)
percentile_95 = simulation_df.quantile(0.95, axis=1)

# ---------------------------------------------------------------
# คำนวณ VaR (เหมือนของเดิม)
# ---------------------------------------------------------------
final_simulated_prices = simulation_df.iloc[-1]
simulated_return_am = (final_simulated_prices / last_price) - 1
cl = user_confidence / 100
percentile = (1 - cl) * 100
VaR_value = np.percentile(simulated_return_am, percentile)

# ---------------------------------------------------------------
# Layout หลัก: รวมกราฟทั้งหมดในหน้าเดียว
# ---------------------------------------------------------------
col1, col2 = st.columns(2)

# ===== กราฟ Histogram ของผลตอบแทนรายวัน =====
with col1:
    st.subheader("📈 Histogram of Daily Return")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.histplot(df_daily_return['Daily Return'] * 100, bins=30,
                 color='skyblue', edgecolor='black', ax=ax1)
    ax1.axvline(mean_return * 100, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_return*100:.2f}%')
    ax1.axvline(mean_return*100 + 2*volatility*100, color='blue', linestyle='--', linewidth=2, label='+2 SD')
    ax1.axvline(mean_return*100 - 2*volatility*100, color='blue', linestyle='--', linewidth=2, label='-2 SD')
    ax1.legend()
    st.pyplot(fig1)

# ===== กราฟ Monte Carlo Simulation =====
with col2:
    st.subheader("📉 Monte Carlo Simulated Price Paths")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(simulation_df, color='red', alpha=0.05)
    ax2.plot(percentile_5, color='green', linestyle='--', label='5th Percentile')
    ax2.plot(percentile_95, color='orange', linestyle='--', label='95th Percentile')
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()
    st.pyplot(fig2)

# ===== กราฟ VaR =====
col3, col4 = st.columns(2)
with col3:
    st.subheader("💥 Value at Risk (VaR)")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.histplot(simulated_return_am * 100, bins=50, color='lightblue', edgecolor='black', ax=ax3)
    ax3.axvline(VaR_value * 100, color='red', linestyle='--', linewidth=2,
                label=f'VaR {user_confidence}%: {VaR_value*100:.2f}%')
    ax3.legend()
    st.pyplot(fig3)

# ===== ตาราง Summary =====
with col4:
    st.subheader("📊 Summary Table")
    summary_df = pd.DataFrame({
        "Metric": ["Mean (μ)", "Volatility (σ)",
                   "Outliers >1SD", "Outliers >2SD", "Outliers >3SD",
                   "5th Percentile Return", "95th Percentile Return",
                   f"VaR {user_confidence}%"],
        "Value": [f"{mean_return*100:.4f}%", f"{volatility*100:.4f}%",
                  f"{out_1sd:.2f}%", f"{out_2sd:.2f}%", f"{out_3sd:.2f}%",
                  f"{(percentile_5.iloc[-1]/last_price - 1)*100:.2f}%",
                  f"{(percentile_95.iloc[-1]/last_price - 1)*100:.2f}%",
                  f"{VaR_value*100:.2f}%"]
    })
    st.table(summary_df)
