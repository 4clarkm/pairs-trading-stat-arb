import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def load_data(start="2010-01-01", end=None):
    """
    Pull adjusted closing prices for V, MA, and SPY.
    Returns a DataFrame of daily prices.
    """
    tickers = ["V", "MA", "SPY"]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    prices = raw.dropna()
    return prices


def compute_signals(prices, window=252, entry_threshold=2.0):
    """
    Computes rolling hedge ratio, spread, z-score, and trading signals
    for the V/MA pairs trade.

    Parameters:
    - prices: DataFrame with at least V and MA columns
    - window: rolling window in days for hedge ratio and z-score
    - entry_threshold: z-score level to enter a trade

    Returns:
    - signals: Series of position signals (1, -1, 0)
    - zscore: Series of rolling z-scores
    - spread: Series of raw spread values
    - rolling_beta: Series of rolling hedge ratios
    - trades: DataFrame of trade log with entry/exit dates and direction
    """
    rolling_beta = pd.Series(index=prices.index, dtype=float)
    rolling_alpha = pd.Series(index=prices.index, dtype=float)

    for i in range(window, len(prices)):
        window_prices = prices.iloc[i-window:i]
        X = add_constant(window_prices["V"])
        y = window_prices["MA"]
        model = OLS(y, X).fit()
        rolling_beta.iloc[i] = model.params["V"]
        rolling_alpha.iloc[i] = model.params["const"]

    spread = prices["MA"] - (rolling_alpha + rolling_beta * prices["V"])
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    zscore = (spread - rolling_mean) / rolling_std

    signals = pd.Series(0.0, index=zscore.index)
    position = 0
    trade_log = []
    entry_date = None
    entry_zscore = None

    for i in range(len(zscore)):
        z = zscore.iloc[i]
        date = zscore.index[i]

        if np.isnan(z):
            signals.iloc[i] = 0
            continue

        if position == 0:
            if z < -entry_threshold:
                position = 1
                entry_date = date
                entry_zscore = z
            elif z > entry_threshold:
                position = -1
                entry_date = date
                entry_zscore = z

        elif position == 1:
            if z >= 0:
                trade_log.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "direction": "long spread",
                    "entry_zscore": round(entry_zscore, 4),
                    "exit_zscore": round(z, 4),
                    "duration_days": (date - entry_date).days
                })
                position = 0

        elif position == -1:
            if z <= 0:
                trade_log.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "direction": "short spread",
                    "entry_zscore": round(entry_zscore, 4),
                    "exit_zscore": round(z, 4),
                    "duration_days": (date - entry_date).days
                })
                position = 0

        signals.iloc[i] = position

    trades = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()

    return signals, zscore, spread, rolling_beta, trades


def compute_returns(prices, signals, idle="cash", cost_per_trade=0.0005):
    """
    Computes daily strategy returns given signals and idle capital choice.

    Parameters:
    - prices: DataFrame with V, MA, SPY columns
    - signals: Series of position signals from compute_signals
    - idle: what to hold when no pairs position - "cash", "ma", "v", or "spy"
    - cost_per_trade: one way transaction cost as a fraction

    Returns:
    - strategy_returns: Series of daily returns
    """
    log_returns = np.log(prices / prices.shift(1))

    # Pairs trade returns when active
    pairs_returns = (
        signals.shift(1) * log_returns["MA"] -
        signals.shift(1) * log_returns["V"]
    )

    # Transaction costs
    position_changes = signals.diff().abs()
    cost_series = position_changes * cost_per_trade * 4

    pairs_returns = pairs_returns - cost_series

    # Idle capital
    if idle == "cash":
        strategy_returns = pairs_returns
    else:
        idle_map = {"ma": "MA", "v": "V", "spy": "SPY"}
        idle_col = idle_map.get(idle.lower(), "MA")
        idle_returns = log_returns[idle_col]

        no_position = signals.shift(1) == 0
        strategy_returns = pairs_returns.copy()
        strategy_returns[no_position] = idle_returns[no_position]

    return strategy_returns.dropna()


def compute_metrics(returns):
    """
    Computes key performance metrics for a return series.

    Parameters:
    - returns: Series of daily returns

    Returns:
    - dict of annualized return, volatility, Sharpe, max drawdown, win rate
    """
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    cumulative = returns.cumsum()
    drawdown = cumulative - cumulative.cummax()
    max_dd = drawdown.min()

    # Win rate - percentage of trading days with positive return
    active_days = returns[returns != 0]
    win_rate = (active_days > 0).sum() / len(active_days) if len(active_days) > 0 else 0

    return {
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate
    }
