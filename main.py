import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Daily ETF Picker", layout="wide")
st.title("Daily ETF Investment Recommendation")

with st.expander("ℹ️ What do the columns mean?"):
    st.markdown("""
    | **Column**             | **Meaning** |
    |------------------------|-------------|
    | **ETF Ticker**         | The NSE ticker symbol of the ETF (e.g., NIFTYBEES.NS). |
    | **ETF Name**           | Full name of the ETF. |
    | **Current Price**      | The latest available closing price of the ETF. |
    | **% Below SMA20**      | How much the ETF's current price is **below its 20-day Simple Moving Average** — helps identify dips. Negative % means it's under the average. |
    | **7-Day Return (%)**   | The percentage price change over the last 7 trading days. Positive = upward momentum. |
    | **RSI**                | The **Relative Strength Index** (0 to 100). < 30 = oversold, > 70 = overbought. |
    | **6M / 1Y / 3Y / 5Y**  | Returns over the past 6 months, 1 year, 3 years, and 5 years. |
    | **Score**              | A custom score combining dip-buying and momentum factors. Higher = better buy. |
    | **Rank**               | Rank based on the Score — **Rank 1 is today's pick**. |
    """)

# List of ETFs with their names
etfs = {
    'NIFTYBEES.NS': 'Nippon India Nifty 50 ETF',
    'ICICINXT50.NS': 'ICICI Prudential Nifty Next 50 ETF',
    'GOLDBEES.NS': 'Nippon India Gold ETF',
    'ITBEES.NS': 'Nippon India ETF IT',
    'BANKBEES.NS': 'Nippon India ETF Bank BeES',
    'MOMENTUM50.NS': 'ICICI Prudential Nifty 200 Momentum 30 ETF',
    'MIDCAPETF.NS': 'Motilal Oswal Midcap 150 ETF',
    'NV20BEES.NS': 'Nippon India ETF NV20',
    'SBIETFQLTY.NS': 'SBI ETF Quality',
    'SBIETFIT.NS': 'SBI ETF IT'
}

# Parameters
sma_window = 20
rsi_period = 14

@st.cache_data
def fetch_etf_data(ticker):
    df = yf.download(ticker, period="30d", interval="1d")
    if df.empty:
        return None
    df['SMA20'] = df['Close'].rolling(window=sma_window).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['7d_return'] = df['Close'].pct_change(periods=7)
    return df

@st.cache_data
def fetch_long_term_returns(ticker):
    df = yf.download(ticker, period="5y", interval="1d")
    if df.empty:
        return None, None, None, None
    
    last_price = df['Close'].iloc[-1]
    returns = {}
    for label, months in [('6M', 6), ('1Y', 12), ('3Y', 36), ('5Y', 60)]:
        date_threshold = datetime.now() - pd.DateOffset(months=months)
        # Get the price closest to the threshold date
        past_prices = df[df.index <= date_threshold]['Close']
        if not past_prices.empty:
            old_price = past_prices.iloc[-1]
            returns[label] = round((last_price - old_price) / old_price * 100, 2)
        else:
            returns[label] = None
    return returns.get('6M'), returns.get('1Y'), returns.get('3Y'), returns.get('5Y')

results = []

# Add refresh button
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()  # Clears all @st.cache_data cached functions
    st.rerun()

for etf, name in etfs.items():
    try:
        data = fetch_etf_data(etf)
        if data is None or data.shape[0] < 21:
            continue
        latest = data.iloc[-1]
        current = float(latest['Close'].item())
        sma = float(latest['SMA20'].item())
        rsi = float(latest['RSI'].item())
        ret7 = float(latest['7d_return'].item())

        if any(np.isnan([sma, rsi, ret7])):
            continue

        ret6m, ret1y, ret3y, ret5y = fetch_long_term_returns(etf)

        # Explicitly cast returns to numeric (float) type
        ret6m = float(ret6m.iloc[0]) if ret6m is not None else None
        ret1y = float(ret1y.iloc[0]) if ret1y is not None else None
        ret3y = float(ret3y.iloc[0]) if ret3y is not None else None
        ret5y = float(ret5y.iloc[0]) if ret5y is not None else None


        # score = (3 * (sma - current) / sma) + (2 * ret7) - (rsi / 100)
        score = (3 * (sma - current) / sma) + (2 * ret7) - (rsi / 100)

        for weight, ret in zip([0.005, 0.005, 0.0025, 0.0025], [ret6m, ret1y, ret3y, ret5y]):
            if ret is not None:
                score += weight * (ret.item() if isinstance(ret, pd.Series) else ret)

        score = float(score)

        results.append({
            'ETF Ticker': etf,
            'ETF Name': name,
            'Current Price': round(current, 2),
            '% Below SMA20': round((sma - current) / sma * 100, 2),
            '7-Day Return (%)': round(ret7 * 100, 2),
            'RSI': round(rsi, 2),
            '6M Return (%)': ret6m,
            '1Y Return (%)': ret1y,
            '3Y Return (%)': ret3y,
            '5Y Return (%)': ret5y,
            'Score': round(score, 3)
        })
    except Exception as e:
        st.warning(f"Failed to fetch data for {etf}: {e}")

if results:
    df = pd.DataFrame(results)
    df.sort_values(by='Score', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Rank'] = df.index + 1

    st.subheader("Today's Ranked ETFs")
    st.dataframe(df.style.highlight_max(axis=0, subset=['Score'], color='lightgreen'))

    top_pick = df.iloc[0]
    st.success(f"Today's recommended ETF to invest ₹1,000 in: **{top_pick['ETF Name']}** ({top_pick['ETF Ticker']})")
else:
    st.warning("No ETF data available or insufficient data to calculate indicators.")