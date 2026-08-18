import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from strategy import load_data, compute_signals, compute_returns, compute_metrics

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="V/MA Pairs Trading Dashboard",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("V/MA Statistical Arbitrage Dashboard")
st.markdown(
    "A market neutral pairs trading strategy exploiting the cointegrated "
    "relationship between Visa (V) and Mastercard (MA). Adjust parameters "
    "in the sidebar to explore how the strategy behaves."
)
st.divider()

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("Strategy Parameters")

window = st.sidebar.slider(
    "Rolling Window (days)",
    min_value=63,
    max_value=504,
    value=252,
    step=63,
    help="Number of days used to estimate the hedge ratio and z-score. 252 = 1 year."
)

entry_threshold = st.sidebar.slider(
    "Entry Threshold (sigma)",
    min_value=1.0,
    max_value=3.0,
    value=2.0,
    step=0.25,
    help="Z-score level required to enter a trade. Higher = fewer but more selective trades."
)

idle_choice = st.sidebar.selectbox(
    "Idle Capital Deployment",
    options=["cash", "ma", "v", "spy"],
    index=3,
    format_func=lambda x: {
        "cash": "Hold Cash",
        "ma": "Hold Mastercard (MA)",
        "v": "Hold Visa (V)",
        "spy": "Hold S&P 500 (SPY)"
    }[x],
    help="What to hold when the pairs signal is inactive."
)

cost_per_trade = st.sidebar.slider(
    "Transaction Cost (bps per side)",
    min_value=0,
    max_value=25,
    value=5,
    step=1,
    help="One-way transaction cost in basis points. 5bps is realistic for liquid large caps."
) / 10000

st.sidebar.divider()
st.sidebar.markdown(
    "**Strategy:** Long MA / Short V when z-score < -threshold. "
    "Short MA / Long V when z-score > +threshold. "
    "Exit when z-score reverts to zero."
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data(start="2010-01-01")

with st.spinner("Loading price data..."):
    prices = get_data()

if prices.empty or len(prices) == 0:
    st.error("Failed to load price data. Please refresh the page.")
    st.stop()

# ── Run strategy ──────────────────────────────────────────────────────────────
with st.spinner("Running strategy..."):
    signals, zscore, spread, rolling_beta, trades = compute_signals(
        prices, window=window, entry_threshold=entry_threshold
    )
    returns = compute_returns(
        prices, signals, idle=idle_choice, cost_per_trade=cost_per_trade
    )
    metrics = compute_metrics(returns)

# Calculate actual returns during stress periods dynamically
covid_return = returns.loc["2020-02-01":"2020-04-30"].sum()
rate_shock_return = returns.loc["2022-01-01":"2022-12-31"].sum()

# ── Metrics panel ─────────────────────────────────────────────────────────────
st.subheader("Performance Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Annualized Return",
    f"{metrics['annualized_return']:.2%}"
)
col2.metric(
    "Annualized Vol",
    f"{metrics['annualized_vol']:.2%}"
)
col3.metric(
    "Sharpe Ratio",
    f"{metrics['sharpe']:.2f}"
)
col4.metric(
    "Max Drawdown",
    f"{metrics['max_drawdown']:.2%}"
)
col5.metric(
    "Win Rate",
    f"{metrics['win_rate']:.2%}"
)

st.divider()

# ── Price history ─────────────────────────────────────────────────────────────
st.subheader("Price History - Visa vs Mastercard")

normalized = prices[["V", "MA"]] / prices[["V", "MA"]].iloc[0] * 100

fig1, ax1 = plt.subplots(figsize=(14, 4))
normalized["V"].plot(ax=ax1, color="steelblue", linewidth=1.2, label="Visa (V)")
normalized["MA"].plot(ax=ax1, color="darkorange", linewidth=1.2, label="Mastercard (MA)")
ax1.set_title("Normalized Price History (Base = 100)", fontsize=12)
ax1.set_ylabel("Normalized Price")
ax1.legend()
ax1.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig1)
plt.close()

st.divider()

# ── Z-score chart with entry/exit markers ─────────────────────────────────────
st.subheader("Z-Score & Trade Signals")

fig2, ax2 = plt.subplots(figsize=(14, 4))
zscore.plot(ax=ax2, color="steelblue", linewidth=1, alpha=0.8, label="Z-Score")
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax2.axhline(entry_threshold, color="crimson", linewidth=1,
            linestyle="--", label=f"+{entry_threshold}σ")
ax2.axhline(-entry_threshold, color="crimson", linewidth=1,
            linestyle="--", label=f"-{entry_threshold}σ")

if len(trades) > 0:
    for _, trade in trades.iterrows():
        color = "green" if trade["direction"] == "long spread" else "red"
        ax2.axvline(trade["entry_date"], color=color, alpha=0.3, linewidth=0.8)

ax2.set_title("Rolling Z-Score with Trade Entries", fontsize=12)
ax2.set_ylabel("Z-Score")
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.divider()

# ── Equity curve ──────────────────────────────────────────────────────────────
st.subheader("Equity Curve")

spy_returns = np.log(prices["SPY"] / prices["SPY"].shift(1)).dropna()
spy_returns = spy_returns.reindex(returns.index)

fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

returns.cumsum().plot(ax=ax3, color="steelblue", linewidth=1.5, label="Strategy")
spy_returns.cumsum().plot(ax=ax3, color="darkorange", linewidth=1.5, label="SPY")

# Shade key stress periods
ax3.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30"),
            alpha=0.15, color="green", label="COVID Crash")
ax3.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
            alpha=0.15, color="purple", label="Rate Shock 2022")

# Dynamic annotations
covid_date = returns.cumsum().index.asof(pd.Timestamp("2020-03-01"))
rate_date = returns.cumsum().index.asof(pd.Timestamp("2022-06-01"))

ax3.annotate(f"COVID\n{covid_return:.1%}", xy=(covid_date,
             returns.cumsum().loc[covid_date] + 0.25),
             fontsize=8, color="green", fontweight="bold")
ax3.annotate(f"Rate Shock\n{rate_shock_return:.1%}", xy=(rate_date,
             returns.cumsum().loc[rate_date] + 0.25),
             fontsize=8, color="purple", fontweight="bold")

ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax3.set_title("Cumulative Log Returns", fontsize=12)
ax3.set_ylabel("Cumulative Log Return")
ax3.legend()
ax3.grid(alpha=0.3)

drawdown = returns.cumsum() - returns.cumsum().cummax()
drawdown.plot(ax=ax4, color="crimson", linewidth=1.5)
ax4.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color="crimson")
ax4.set_title("Drawdown", fontsize=12)
ax4.set_ylabel("Drawdown")
ax4.grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig3)
plt.close()

st.divider()

# ── Trades table ──────────────────────────────────────────────────────────────
st.subheader("Trade Log")

if len(trades) > 0:
    st.dataframe(
        trades.style.format({
            "entry_zscore": "{:.4f}",
            "exit_zscore": "{:.4f}",
            "duration_days": "{:.0f}"
        }),
        use_container_width=True
    )
    st.caption(
        f"Total trades: {len(trades)} - "
        f"Avg duration: {trades['duration_days'].mean():.1f} days - "
        f"Long spread: {(trades['direction'] == 'long spread').sum()} - "
        f"Short spread: {(trades['direction'] == 'short spread').sum()}"
    )
else:
    st.info("No completed trades with current parameters.")

st.divider()

# ── Rolling hedge ratio ────────────────────────────────────────────────────────
st.subheader("Rolling Hedge Ratio")

fig4, ax5 = plt.subplots(figsize=(14, 3))
rolling_beta.plot(ax=ax5, color="steelblue", linewidth=1.2)
ax5.set_title("Rolling Hedge Ratio (β) - MA = α + β·V + ε", fontsize=12)
ax5.set_ylabel("β")
ax5.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig4)
plt.close()

st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "Built by Mason Clark - UW-Madison Quantitative Economics & Mathematics | "
    "[GitHub](https://github.com/4clarkm/pairs-trading-stat-arb)"
)
