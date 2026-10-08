# FTSE 350 Equity Screener

A Python tool that screens FTSE 350 stocks in real time using valuation multiples and price momentum signals.

## What it does

- Pulls live price and fundamental data via Yahoo Finance API
- Filters stocks by P/E, P/B, EV/EBITDA, dividend yield, and market cap
- Calculates 3-month, 6-month, and 12-month price momentum for each stock
- Exports results to a dated CSV file

## Preset screens

| Screen | Criteria |
|--------|----------|
| Value | P/E ≤ 14, P/B ≤ 2, positive 12m momentum |
| Income | Dividend yield ≥ 4%, P/E ≤ 20, 6m momentum > 0 |
| Momentum | 3m > 5%, 6m > 8%, 12m > 10% |
| Quality-Value | P/E ≤ 18, P/B ≤ 3, EV/EBITDA ≤ 12, yield ≥ 2%, 6m momentum > 0, market cap ≥ £1bn |

## Usage

```bash
pip install yfinance pandas numpy
python ftse350_equity_screener.py
```

Or customise thresholds directly:

```python
from ftse350_equity_screener import run_screener
results = run_screener(max_pe=15, min_div_yield=0.03, min_momentum_3m=0.05)
```

## Methodology

Momentum is calculated as trailing price returns over 63, 126, and 252 trading days (approximate 3, 6, and 12 month windows). Valuation data is sourced from Yahoo Finance's fundamentals feed. Universe is a hand-picked sample of 196 LSE-listed FTSE 350 constituents (not the full index), using `.L` ticker suffixes. Dividend yields are normalised to fractions because yfinance has returned both formats.
