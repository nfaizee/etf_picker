import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64

st.set_page_config(page_title="Daily ETF Picker", layout="wide")
st.title("Daily ETF Investment Recommendation")

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

# Default parameters
DEFAULT_PARAMS = {
    'sma_window': 20,
    'rsi_period': 14,
    'sma_weight': 3,
    'return_weight': 2,
    'rsi_weight': 1,
    'transaction_fee': 0,
}

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

# Parameters
sma_window = DEFAULT_PARAMS['sma_window']
rsi_period = DEFAULT_PARAMS['rsi_period']

# Add tabs for the main app and backtesting
tab1, tab2 = st.tabs(["Daily ETF Picker", "Backtest Strategy"])

##################
# ETF PICKER TAB #
##################

with tab1:
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

    # Add refresh button
    if st.button("🔄 Refresh Data", key="refresh_daily"):
        st.cache_data.clear()  # Clears all @st.cache_data cached functions
        st.rerun()

    results = []

    for etf, name in etfs.items():
        try:
            data = fetch_etf_data(etf)
            if data is None or data.shape[0] < 21:
                continue
            latest = data.iloc[-1]
            def safe_float(val):
                if isinstance(val, pd.Series):
                    return float(val.iloc[0])
                return float(val)
            current = safe_float(latest['Close'])
            sma = safe_float(latest['SMA20'])
            rsi = safe_float(latest['RSI'])
            ret7 = safe_float(latest['7d_return'])

            if any(np.isnan([sma, rsi, ret7])):
                continue

            ret6m, ret1y, ret3y, ret5y = fetch_long_term_returns(etf)
            
            # Convert returns to proper format for calculation
            def safe_float(val):
                if val is None:
                    return None
                if isinstance(val, pd.Series):
                    return float(val.iloc[0])
                return float(val)

            ret6m = safe_float(ret6m)
            ret1y = safe_float(ret1y)
            ret3y = safe_float(ret3y)
            ret5y = safe_float(ret5y)

            # Calculate score
            score = (DEFAULT_PARAMS['sma_weight'] * (sma - current) / sma) + \
                    (DEFAULT_PARAMS['return_weight'] * ret7) - \
                    (DEFAULT_PARAMS['rsi_weight'] * (rsi / 100))

            # Add historical returns to score with small weights
            for weight, ret in zip([0.005, 0.005, 0.0025, 0.0025], [ret6m, ret1y, ret3y, ret5y]):
                if ret is not None:
                    score += weight * ret

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

#####################
# BACKTESTING TAB  #
#####################

with tab2:
    st.header("Backtest ETF Strategy")
    
    # Create two columns for input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        backtest_period = st.selectbox(
            "Backtest Period",
            options=["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "max"],
            index=3,
            help="Time period to run the backtest"
        )
        
        daily_investment = st.number_input(
            "Daily Investment Amount (₹)",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Amount to invest daily in the selected ETF"
        )
        
    with col2:
        transaction_fee = st.number_input(
            "Transaction Fee per Trade (₹)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            help="Fee charged per transaction"
        )
        
        run_sensitivity = st.checkbox(
            "Run Parameter Sensitivity Analysis", 
            value=False,
            help="Test different parameter combinations to find optimal settings"
        )
    
    st.markdown("---")
    
    # Button to start the backtest
    start_backtest = st.button("🚀 Run Backtest", key="run_backtest")
    
    # Backtesting functions
    def fetch_all_data(period='1y'):
        """Download data for all ETFs and calculate technical indicators"""
        with st.spinner("Downloading ETF data..."):
            progress_bar = st.progress(0)
            data = {}
            
            for i, (ticker, name) in enumerate(etfs.items()):
                try:
                    progress_bar.progress((i + 1) / len(etfs))
                    st.write(f"Downloading {name} ({ticker})...")
                    
                    df = yf.download(ticker, period=period, interval='1d', progress=False, auto_adjust=True)
                    
                    if df.empty:
                        st.warning(f"⚠️ No data found for {ticker}")
                        continue
                        
                    # Calculate indicators
                    df['SMA20'] = df['Close'].rolling(window=DEFAULT_PARAMS['sma_window']).mean()
                    
                    # RSI calculation
                    delta = df['Close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(window=DEFAULT_PARAMS['rsi_period']).mean()
                    loss = -delta.where(delta < 0, 0).rolling(window=DEFAULT_PARAMS['rsi_period']).mean()
                    rs = gain / loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                    
                    # Returns over different periods
                    df['1d_return'] = df['Close'].pct_change(periods=1)
                    df['7d_return'] = df['Close'].pct_change(periods=7)
                    df['30d_return'] = df['Close'].pct_change(periods=30)
                    
                    # Add to data dictionary
                    data[ticker] = df
                    
                except Exception as e:
                    st.error(f"❌ Error downloading {ticker}: {e}")
            
            progress_bar.empty()
            
            if not data:
                st.error("No data was fetched for any ETFs. Check your internet connection or ETF tickers.")
                return None
            
            return data
    
    def score_etf(close, sma, ret7, rsi, params=None):
        """Calculate score for an ETF based on technical indicators"""
        if params is None:
            params = DEFAULT_PARAMS
        
        if pd.isna(close) or pd.isna(sma) or pd.isna(ret7) or pd.isna(rsi):
            return -np.inf
            
        sma_component = params['sma_weight'] * (sma - close) / sma
        return_component = params['return_weight'] * ret7
        rsi_component = -params['rsi_weight'] * (rsi / 100)
        
        return sma_component + return_component + rsi_component
    
    def simulate_buy_and_hold(data, daily_investment):
        """Simulate buying and holding equal amounts of each ETF"""
        all_dates = sorted(set(date for df in data.values() for date in df.index))
        start_date = all_dates[0]
        end_date = all_dates[-1]
        
        total_invested = 0
        holdings = {}
        
        # Calculate number of trading days
        trading_days = len(all_dates)
        
        # Split investment equally among all ETFs
        per_etf_investment = daily_investment / len(data)
        
        for ticker, df in data.items():
            if start_date in df.index and end_date in df.index:
                start_price = safe_float(df.loc[start_date, 'Close'])
                end_price = safe_float(df.loc[end_date, 'Close'])
                
                # Total investment in this ETF over all days
                total_etf_investment = per_etf_investment * trading_days
                total_invested += total_etf_investment
                
                # Units bought
                units = total_etf_investment / start_price
                
                # Final value
                final_value = units * end_price
                
                holdings[ticker] = {
                    'units': units,
                    'invested': total_etf_investment,
                    'final_value': final_value,
                    'return': (final_value / total_etf_investment - 1) * 100
                }
        
        # Calculate total portfolio value
        final_value = sum(h['final_value'] for h in holdings.values())
        
        return {
            'total_invested': float(total_invested),
            'final_value': float(final_value),
            'profit': float(final_value - total_invested),
            'return': float((final_value / total_invested - 1) * 100),
            'holdings': holdings
        }
    
    def simulate_strategy(data, params=None):
        """Simulate the ETF selection strategy"""
        if params is None:
            params = DEFAULT_PARAMS.copy()
        
        daily_investment = params.get('daily_investment', 1000)
        fee = params.get('transaction_fee', 0)
        
        all_dates = sorted(set(date for df in data.values() for date in df.index))
        portfolio = []
        daily_scores = {}

        for date in all_dates:
            scores = []
            for ticker, df in data.items():
                if date not in df.index:
                    continue
                
                # Skip if we don't have enough history for indicators
                if date < df.index[20]:  # Need at least 20 days for SMA20
                    continue

                # Get values for scoring
                try:
                    def safe_float(val):
                        if isinstance(val, pd.Series):
                            return float(val.iloc[0])
                        return float(val)

                    close = safe_float(df.loc[date, 'Close'])
                    sma = safe_float(df.loc[date, 'SMA20'])
                    rsi = safe_float(df.loc[date, 'RSI'])
                    ret7 = safe_float(df.loc[date, '7d_return'])
                    
                    if pd.isna(close) or pd.isna(sma) or pd.isna(rsi) or pd.isna(ret7):
                        continue
                        
                    score = score_etf(close, sma, ret7, rsi, params)
                    scores.append((ticker, score, close))
                except Exception:
                    continue

            # Store all scores for analysis
            daily_scores[date] = sorted(scores, key=lambda x: x[1], reverse=True)
            
            if scores:
                best = max(scores, key=lambda x: x[1])
                best_ticker, best_score, close_price = best

                if pd.notna(close_price) and close_price > 0:
                    # Apply transaction fee
                    net_investment = daily_investment - fee
                    if net_investment <= 0:
                        continue
                        
                    units = net_investment / close_price
                    portfolio.append({
                        'date': date,
                        'ticker': best_ticker,
                        'etf_name': etfs[best_ticker],
                        'price': close_price,
                        'units': units,
                        'investment': daily_investment,
                        'net_investment': net_investment,
                        'score': best_score
                    })

        portfolio_df = pd.DataFrame(portfolio)
        if portfolio_df.empty:
            st.error("No investments were made. Check data availability.")
            return None

        # Calculate cumulative investment over time
        cumulative = portfolio_df.groupby('date').agg({'investment': 'sum'}).cumsum()
        
        # Calculate cumulative units held by ticker
        units_by_ticker = portfolio_df.groupby(['date', 'ticker'])['units'].sum().unstack().fillna(0).cumsum()

        # Get final prices for each ETF
        final_prices = {}
        for ticker in units_by_ticker.columns:
            ticker_data = data.get(ticker)
            if ticker_data is not None and not ticker_data.empty:
                final_price = ticker_data['Close'].iloc[-1]
                final_price = final_price if not isinstance(final_price, pd.Series) else final_price.iloc[0] 
                final_prices[ticker] = float(final_price)
            else:
                final_prices[ticker] = 0

        # Calculate final portfolio value
        ticker_values = {}
        for ticker in units_by_ticker.columns:
            final_units = float(units_by_ticker.iloc[-1][ticker])
            final_price = final_prices[ticker]
            
            ticker_value = final_units * final_price
            ticker_values[ticker] = {
                'units': final_units,
                'price': final_price,
                'value': ticker_value,
                'name': etfs[ticker]
            }

        # Calculate final portfolio value
        final_value = sum(info['value'] for info in ticker_values.values())

        # Convert to Python float to avoid Series conversion issues
        total_invested = float(cumulative.iloc[-1]['investment']) if not cumulative.empty else 0
        total_fees = float(portfolio_df.shape[0] * fee)
        net_invested = total_invested - total_fees
        profit = final_value - net_invested

        # Calculate ticker distribution in final portfolio
        for ticker, info in ticker_values.items():
            if final_value > 0:
                info['pct_of_portfolio'] = (info['value'] / final_value * 100)
            else:
                info['pct_of_portfolio'] = 0

        # Calculate investment frequency for each ETF
        investment_counts = portfolio_df['ticker'].value_counts()
        for ticker in ticker_values:
            ticker_values[ticker]['times_selected'] = investment_counts.get(ticker, 0)
            ticker_values[ticker]['selection_pct'] = (investment_counts.get(ticker, 0) / len(portfolio_df) * 100)

        return {
            'cumulative': cumulative,
            'portfolio': portfolio_df,
            'total_invested': total_invested,
            'total_fees': total_fees,
            'net_invested': net_invested,
            'final_value': final_value,
            'profit': profit,
            'return_pct': (profit / net_invested * 100) if net_invested > 0 else 0,
            'holdings': ticker_values,
            'daily_scores': daily_scores
        }
    
    def plot_results(result, buy_hold_result=None, data=None):
        """Plot the strategy performance and return the figure"""
        # 1. Portfolio Value vs Investment plot
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot total invested amount
        ax1.plot(result['cumulative'].index, result['cumulative']['investment'], 
                label='Total Invested (₹)', color='blue')
        
        # Calculate portfolio value over time
        portfolio_df = result['portfolio']
        all_dates = sorted(portfolio_df['date'].unique())
        portfolio_values = []
        
        for date in all_dates:
            date_value = 0
            # For each ticker, calculate the value of holdings on this date
            for ticker, holdings_info in result['holdings'].items():
                # Get units held on this date
                units_held = portfolio_df[(portfolio_df['date'] <= date) & (portfolio_df['ticker'] == ticker)]['units'].sum()
                # Get price on this date
                for etf_ticker, etf_data in data.items():
                    if etf_ticker == ticker and date in etf_data.index:
                        price = safe_float(etf_data.loc[date, 'Close'])
                        date_value += units_held * price
                        break
            
            portfolio_values.append((date, date_value))
        
        if portfolio_values:  # Check if list is not empty
            dates, values = zip(*portfolio_values)
            ax1.plot(dates, values, label='Portfolio Value (₹)', color='green')
        
        # Add reference line for final value
        ax1.axhline(y=float(result['final_value']), color='green', linestyle='--', 
                    label=f'Final Value: ₹{float(result["final_value"]):,.2f}')
        
        # Add buy & hold reference if available
        if buy_hold_result:
            ax1.axhline(y=float(buy_hold_result['final_value']), color='red', linestyle='--', 
                        label=f'Buy & Hold Value: ₹{float(buy_hold_result["final_value"]):,.2f}')
        
        ax1.set_title("ETF Strategy Performance")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Value (₹)")
        ax1.legend()
        ax1.grid(True)
        fig1.tight_layout()
        
        # 2. ETF Selection frequency plot
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        holdings = result['holdings']
        tickers = [ticker for ticker in holdings.keys()]
        selection_counts = [holdings[ticker]['times_selected'] for ticker in tickers]
        
        # Sort by selection frequency
        sorted_indices = np.argsort(selection_counts)[::-1]
        sorted_tickers = [tickers[i] for i in sorted_indices]
        sorted_counts = [selection_counts[i] for i in sorted_indices]
        
        ax2.bar(sorted_tickers, sorted_counts)
        ax2.set_title("ETF Selection Frequency")
        ax2.set_xlabel("ETF")
        ax2.set_ylabel("Number of Days Selected")
        plt.xticks(rotation=45)
        fig2.tight_layout()
        
        return fig1, fig2
    
    def run_parameter_sensitivity(data, param_ranges):
        """Run strategy with different parameters to find optimal settings"""
        results = []
        
        # Create baseline parameters
        base_params = DEFAULT_PARAMS.copy()
        base_params['daily_investment'] = daily_investment
        base_params['transaction_fee'] = transaction_fee
        
        progress_bar = st.progress(0)
        total_tests = sum(len(values) for values in param_ranges.values())
        test_count = 0
        
        # For each parameter to test
        for param_name, param_values in param_ranges.items():
            for value in param_values:
                # Create a new parameter set with just this change
                test_params = base_params.copy()
                test_params[param_name] = value
                
                # Run simulation with these parameters
                try:
                    result = simulate_strategy(data, test_params)
                    if result:
                        results.append({
                            'param_name': param_name,
                            'param_value': value,
                            'return_pct': float(result['return_pct']),
                            'profit': float(result['profit'])
                        })
                        st.write(f"Tested {param_name}={value}: Return={float(result['return_pct']):.2f}%, Profit=₹{float(result['profit']):,.2f}")
                except Exception as e:
                    st.warning(f"Error testing {param_name}={value}: {e}")
                
                # Update progress bar
                test_count += 1
                progress_bar.progress(test_count / total_tests)
        
        # Convert to DataFrame for easy analysis
        sensitivity_df = pd.DataFrame(results)
        
        # Create sensitivity plot if we have results
        if not sensitivity_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for param_name in param_ranges.keys():
                param_data = sensitivity_df[sensitivity_df['param_name'] == param_name]
                if not param_data.empty:
                    ax.plot(param_data['param_value'], param_data['return_pct'], 
                            marker='o', label=f"{param_name}")
            
            ax.set_title("Parameter Sensitivity Analysis")
            ax.set_xlabel("Parameter Value")
            ax.set_ylabel("Return (%)")
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            
            # Find optimal parameters
            if not sensitivity_df.empty and not sensitivity_df['return_pct'].empty:
                best_idx = sensitivity_df['return_pct'].idxmax()
                if pd.notna(best_idx):
                    best_result = sensitivity_df.loc[best_idx]
                    st.success(f"Best parameter found: {best_result['param_name']}={best_result['param_value']} "
                            f"with return of {best_result['return_pct']:.2f}%")
            
            return fig, sensitivity_df
        
        return None, sensitivity_df
    
    def print_results(result, buy_hold_result=None):
        """Format and display results in Streamlit"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Strategy Performance")
            st.metric("Total Invested", f"₹{float(result['total_invested']):,.2f}")
            st.metric("Transaction Fees", f"₹{float(result['total_fees']):,.2f}")
            st.metric("Net Invested", f"₹{float(result['net_invested']):,.2f}")
            st.metric("Final Portfolio Value", f"₹{float(result['final_value']):,.2f}")
            st.metric("Profit", f"₹{float(result['profit']):,.2f}", 
                    delta=f"{float(result['return_pct']):.2f}%")
        
        if buy_hold_result:
            with col2:
                st.subheader("Buy & Hold Comparison")
                st.metric("Buy & Hold Invested", f"₹{float(buy_hold_result['total_invested']):,.2f}")
                st.metric("Buy & Hold Final Value", f"₹{float(buy_hold_result['final_value']):,.2f}")
                st.metric("Buy & Hold Profit", f"₹{float(buy_hold_result['profit']):,.2f}", 
                        delta=f"{float(buy_hold_result['return']):,.2f}%")
                
                outperformance = result['return_pct'] - buy_hold_result['return']
                st.metric("Strategy Outperformance", f"{outperformance:.2f}%")
        
        st.subheader("Final Portfolio Composition")
        
        # Create portfolio composition dataframe
        holdings_data = []
        for ticker, info in sorted(result['holdings'].items(), key=lambda x: x[1]['value'], reverse=True):
            if info['units'] > 0:
                holdings_data.append({
                    "Ticker": ticker,
                    "ETF Name": etfs[ticker],
                    "Units": f"{info['units']:.2f}",
                    "Final Price": f"₹{info['price']:.2f}",
                    "Value": f"₹{info['value']:,.2f}",
                    "% of Portfolio": f"{info['pct_of_portfolio']:.2f}%",
                    "Days Selected": f"{info['times_selected']} ({info['selection_pct']:.1f}%)"
                })
        
        if holdings_data:
            holdings_df = pd.DataFrame(holdings_data)
            st.dataframe(holdings_df, use_container_width=True)
    
    # Run the backtest when the button is clicked
    if start_backtest:
        # Set parameters for backtest
        backtest_params = DEFAULT_PARAMS.copy()
        backtest_params['daily_investment'] = daily_investment
        backtest_params['transaction_fee'] = transaction_fee
        
        # Fetch data
        etf_data = fetch_all_data(period=backtest_period)
        
        if etf_data:
            with st.spinner("Running backtest simulation..."):
                # Run strategy simulation
                result = simulate_strategy(etf_data, backtest_params)
                
                # Run buy & hold benchmark
                buy_hold_result = simulate_buy_and_hold(etf_data, daily_investment)
                
                if result:
                    # Print results
                    print_results(result, buy_hold_result)
                    
                    # Plot results
                    st.subheader("Performance Charts")
                    fig1, fig2 = plot_results(result, buy_hold_result, etf_data)
                    st.pyplot(fig1)
                    st.pyplot(fig2)
                    
                    # Run parameter sensitivity analysis if requested
                    if run_sensitivity:
                        st.subheader("Parameter Sensitivity Analysis")
                        st.info("Testing different parameter values to find optimal settings...")
                        
                        # Define ranges for parameters to test
                        param_ranges = {
                            'sma_window': [5, 10, 15, 20, 25, 30],
                            'rsi_period': [7, 10, 14, 20],
                            'sma_weight': [1, 2, 3, 4, 5],
                            'return_weight': [1, 2, 3, 4],
                            'rsi_weight': [0.5, 1, 1.5, 2]
                        }
                        
                        sensitivity_fig, sensitivity_results = run_parameter_sensitivity(etf_data, param_ranges)
                        if sensitivity_fig:
                            st.pyplot(sensitivity_fig)
                        if not sensitivity_results.empty:
                            st.dataframe(sensitivity_results)