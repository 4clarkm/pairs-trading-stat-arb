# Pairs Trading & Statistical Arbitrage
### V/MA Cross-Sectional Mean Reversion Strategy

This project represents my first serious attempt at building and backtesting a quantitative trading strategy from scratch. The goal was not just to make something that looked good on paper, but to build something rigorous, with proper statistical testing, realistic transaction costs, and honest evaluation of limitations. There is also an interactive dashboard hosted by streamlit below to help visualize some of the final results. on

---

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mclark-pairs-trading.streamlit.app/)

**[Live Dashboard](https://mclark-pairs-trading.streamlit.app/)** | [GitHub Repository](https://github.com/4clarkm/pairs-trading-stat-arb)

## Background and Motivation

I chose pairs trading as my entry point into quantitative finance because it sits at the intersection of econometrics, statistics, and market microstructure, all areas I find genuinely interesting. The strategy is market neutral by construction, meaning it does not bet on the market going up or down but instead exploits temporary mispricings between two highly related assets. This makes it a fundamentally different kind of strategy from buy-and-hold or momentum investing, and one that has real portfolio value precisely because of its low correlation to the broader market.

Visa and Mastercard were selected as the trading pair because they share nearly identical business models, payment network duopolies with no direct credit risk on their balance sheets, the same regulatory environment, and the same fundamental driver: consumer spending volumes. The economic case for a stable long run relationship between their prices is airtight, which is exactly what a pairs trading strategy requires.

---

## Methodology

### Cointegration Testing
Before building any strategy, we formally tested whether V and MA are cointegrated using the Engle-Granger two-step method. This involves regressing one price series on the other, then testing the residuals for stationarity using the Augmented Dickey-Fuller test. The null hypothesis is that the spread has a unit root, that it wanders randomly with no tendency to revert. We rejected the null at the 1% significance level (p-value = 0.004), confirming the pair is cointegrated and the spread is stationary and mean reverting.

This step is critical. Without formal cointegration testing, any apparent relationship between two trending price series could be spurious - two independent random walks that happen to move together by chance.

### Rolling Hedge Ratio
Rather than estimating a single fixed hedge ratio over the full sample, we re-estimate the hedge ratio each day using a rolling 252-day window. This ensures every trading decision uses only information available at that point in time, eliminating lookahead bias. The hedge ratio tells us how much MA moves for every unit move in V, and allows us to construct a spread that is comparable across time as the relationship between the two stocks evolves.

### Z-Score Signal Construction
Each day we compute the spread between MA and the hedge-ratio-adjusted V price, then convert it to a z-score using the rolling mean and standard deviation. The z-score measures how many standard deviations the spread is from its historical equilibrium. We enter a long spread position when the z-score falls below -2, a short spread position when it rises above +2, and exit when it reverts to zero.

### Parameter Optimization and Robustness Testing
We tested the strategy across a range of rolling windows (63, 126, 252, and 504 days) and entry thresholds (1.5, 2.0, 2.5, and 3.0 sigma). For the base strategy holding cash when inactive, performance improved monotonically toward longer windows and lower thresholds - a boundary result we treated with skepticism as a potential sign of overfitting. For the MA overlay strategy, we found a genuine interior optimum at 252 days where performance degraded in both directions, giving us confidence the parameter choice is not overfit to the historical data.

### Transaction Costs
We modeled transaction costs as a fixed per-side percentage applied to both legs on entry and exit. At realistic costs of 5 basis points per side - appropriate for liquid large cap equities like V and MA, the strategy remains viable with a positive Sharpe ratio. At 25 basis points the strategy breaks down, establishing a clear cost threshold for deployment.

### Idle Capital Deployment
The pairs signal is inactive approximately 65% of trading days. We tested four alternatives for deploying idle capital: holding cash, SPY, V, or MA. Holding MA during inactive periods produced the strongest result, 22.32% annualized return and 0.96 Sharpe, and genuinely outperformed simply holding MA for the full period (19.31% return, 0.72 Sharpe, -52.76% max drawdown). The pairs signal adds real value as a drawdown buffer during MA's worst periods.

---

## Key Results

| Strategy | Ann. Return | Sharpe | Max Drawdown |
|---|---|---|---|
| Base (cash when idle) | 2.64% | 0.34 | -13.44% |
| Hold MA when idle | 22.32% | 0.96 | -32.99% |
| Hold MA entire period | 19.31% | 0.72 | -52.76% |
| S&P 500 (SPY) | 13.17% | 0.77 | -41.12% |

### Why This Strategy Shines During Market Crises
The most important property of a market neutral strategy is not its average return, it is its behavior when markets break down. The COVID crash and 2022 rate shock results illustrate this directly.

During COVID (February - April 2020), consumer spending collapsed overnight. Both Visa and Mastercard were hit by the same demand shock simultaneously, keeping the spread stable while long only portfolios lost 30%+ in weeks. The strategy returned +10.9% with a 3.90 Sharpe and only -1.83% max drawdown during one of the fastest selloffs in market history.

During the 2022 rate shock, rising interest rates compressed equity valuations across the board. Again both payment networks faced those issues, preserving the spread relationship while the S&P 500 fell roughly 20%. The strategy returned +5.33% for the year.

This is not luck, it is the structural consequence of trading two economically identical businesses. Systemic shocks affect both sides of the trade equally and cancel out, leaving only spread risk which is small and mean reverting by design.

### Stress Test Highlights
The strategy's most important property is its behavior during market crises:

| Period | Total Return | Sharpe |
|---|---|---|
| COVID Crash (Feb-Apr 2020) | +10.90% | 3.90 |
| Rate Shock (2022) | +5.33% | 0.75 |
| Q4 Selloff (2018) | -2.57% | -0.86 |

The COVID and 2022 results are the core of the strategy's value proposition. When the market is in crisis, V and MA are hit by the same shocks simultaneously, keeping the spread stable or creating better trading opportunities while long only portfolios suffer.

---

## Strategic Framing

This is not a strategy designed to beat the market on raw returns. It is a market neutral return stream that holds its value or profits during periods when most other strategies suffer most. Its real value is as a complement to a more aggressive primary strategy - a portfolio stabilizer that earns during the periods a trend following or long only strategy bleeds.

This is the same principle behind dedicated stat arb desks at hedge funds: not the primary return driver, but the hedge that keeps overall fund performance stable during geopolitical shocks, rate regime changes, or liquidity crises.

---

## Project Structure
pairs-trading-stat-arb/

notebooks/
01_data_collection.ipynb
02_cointegration_testing.ipynb
03_pairs_trading_strategy.ipynb
04_multi_pair_portfolio.ipynb
05_robustness_testing.ipynb
06_transaction_costs.ipynb
07_monte_carlo_stress_testing.ipynb
src/
strategy.py
data/
README.md

---

## Limitations and Future Work

- The backtest covers 2010 to present, a period dominated by a historic bull market. Bear regime behavior is underrepresented in the sample.
- The hedge ratio is estimated using OLS which assumes a linear and stable relationship. A Kalman filter would allow the hedge ratio to update dynamically in real time.
- Transaction costs are modeled as fixed and symmetric. In practice costs vary by trade size and market conditions.
- Only one pair passed formal cointegration testing across the candidates tested. A larger universe with systematic pair selection using the distance method would be a natural extension.
- Monte Carlo simulation assumes the future return distribution resembles the historical one. Structural breaks - a merger between V and MA or major regulatory disruption - would invalidate the simulation.

**Next steps:** Kalman filter hedge ratio, expanded universe with systematic pair selection, machine learning signal filter to estimate trade success probability, and an interactive Streamlit dashboard for live parameter exploration.

---

## Tools and Libraries

Python, pandas, NumPy, statsmodels, yfinance, matplotlib, seaborn, scipy
