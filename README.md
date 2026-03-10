# S-Curve-Model
This is a very strict Gompertz/Bass stock screener, 99% of the time no stock selection.  
If you want one rule: run every Friday after market close, and run an extra cycle during peak earnings weeks.
2 Choices: Fast and Normal.  Fast Takes about 3 minutes to complete.  Normal Takes about an hour

Builds a filtered stock universe (S&amp;P500+Nasdaq), pulls yfinance fundamentals/market data, computes S-curve fits (Gompertz/Bass) when possible, otherwise no selection, .

Output is 4 files: scores.csv (ranked names + scores), fit_diagnostics.csv (fit pass/fail reasons), portfolio_weights.csv (final holdings/weights), and summary.json (run stats).
Stocks are selected by highest score_total after universe filters, policy/risk exclusions, and top-percentile cutoff.

Fast = faster iteration, useful for debugging and strategy testing.
Normal = more complete research output, likely longer but better coverage and fidelity.

Model Risks

Data quality risk: yfinance gaps/latency can distort ranks.
Fit risk: many names fail S-curve fit (too_few_points), so 99% of the time no stock selection.
Regime risk: growth signals can underperform in value/defensive markets.
Concentration risk: top-percentile picks may cluster by theme/factor.
Valuation trap risk: “high growth” can still be overpriced or low quality.
Execution risk: slippage, spread, liquidity, tax, and turnover drag.
Model risk: thresholds/weights are hand-tuned and can be unstable.


Treat output as a shortlist, not a buy list.
Check each name’s earnings trend, guidance, and balance sheet.
Validate valuation (P/E, EV/FCF, margins, FCF quality) vs peers.
Read filings/transcripts for business risks and accounting quality.
Enforce personal rules: max position size, stop loss, cash buffer.
Diversify across sectors/factors; avoid theme overexposure.
Paper trade first; track hit-rate, drawdown, and turnover.
Rebalance on a fixed schedule and avoid emotional overrides.
Exclude names/businesses outside your mandate (country/crypto/etc.).
Never allocate capital you can’t afford to lose.



