"""
FTSE 350 Equity Screener
========================
Filters FTSE 350 stocks by valuation multiples and momentum signals.

Author: Roman Falla
GitHub: github.com/roman-falla

Dependencies:
    pip install yfinance pandas numpy

Usage:
    python ftse350_equity_screener.py

    Or import and customise thresholds:
        from ftse350_equity_screener import run_screener
        results = run_screener(max_pe=20, min_momentum_3m=0.05)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# FTSE 350 TICKERS
# A representative sample of FTSE 350 constituents (London Stock Exchange)
# Tickers use the Yahoo Finance format (suffix .L for LSE-listed stocks)
# ─────────────────────────────────────────────────────────────────────────────
FTSE_350_TICKERS = [
    # FTSE 100
    "AAL.L", "ABF.L", "ADM.L", "AHT.L", "ANTO.L", "AZN.L", "AUTO.L",
    "AV.L", "BA.L", "BARC.L", "BATS.L", "BEZ.L", "BP.L", "BRBY.L",
    "BT.A.L", "CCH.L", "CNA.L", "CPG.L", "CRDA.L", "DCC.L", "DGE.L",
    "DPLM.L", "EDV.L", "ENT.L", "EXPN.L", "EZJ.L", "FCIT.L", "FLTR.L",
    "FRES.L", "GLEN.L", "GSK.L", "HIK.L", "HL.L", "HLMA.L", "HLN.L",
    "HSBA.L", "IAG.L", "ICP.L", "IGG.L", "IMB.L", "INF.L", "ITRK.L",
    "JD.L", "KGF.L", "LAND.L", "LGEN.L", "LLOY.L", "LMP.L", "LSE.L",
    "MNDI.L", "MNG.L", "MRO.L", "NG.L", "NWG.L", "NXT.L", "OCDO.L",
    "PCT.L", "PHNX.L", "PRU.L", "PSH.L", "PSN.L", "PSON.L", "RB.L",
    "RDSA.L", "REL.L", "RIO.L", "RKT.L", "RMV.L", "RR.L", "RS1.L",
    "RTO.L", "SBRY.L", "SDR.L", "SGE.L", "SGRO.L", "SHEL.L", "SKG.L",
    "SMDS.L", "SMIN.L", "SMT.L", "SN.L", "SPX.L", "SSE.L", "STAN.L",
    "SVT.L", "TSCO.L", "TW.L", "ULVR.L", "UU.L", "VOD.L", "WEIR.L",
    "WPP.L", "WTB.L",
    # FTSE 250 (selection)
    "ABG.L", "ACSO.L", "AGK.L", "AML.L", "BNZL.L", "BOWL.L", "BTG.L",
    "CAL.L", "CASH.L", "CBG.L", "CINE.L", "CLG.L", "CMC.L", "COB.L",
    "CTEC.L", "CVS.L", "DARK.L", "DNLM.L", "DTY.L", "DWF.L", "ECM.L",
    "EMG.L", "ENTR.L", "ESNT.L", "FDM.L", "FGP.L", "FLTK.L", "FSV.L",
    "GNC.L", "GPOR.L", "GRI.L", "GRG.L", "GTLS.L", "HAT.L", "HBR.L",
    "HFD.L", "HMSO.L", "HUW.L", "HWDN.L", "IHG.L", "IMI.L", "INCH.L",
    "ITV.L", "JET2.L", "JUP.L", "KIE.L", "LAD.L", "LIO.L", "LRE.L",
    "MCS.L", "MERI.L", "MGAM.L", "MKS.L", "MNKS.L", "MPI.L", "MRW.L",
    "MTM.L", "MTO.L", "MWE.L", "NCT.L", "NETW.L", "NMC.L", "NXRT.L",
    "OSB.L", "OXB.L", "PAG.L", "PCA.L", "PETS.L", "PFC.L", "PMVD.L",
    "PNN.L", "POLR.L", "PZC.L", "QQ.L", "RDW.L", "RGD.L", "RHI.L",
    "RHIM.L", "RNK.L", "SAFE.L", "SCT.L", "SHI.L", "SLA.L", "SLP.L",
    "SNN.L", "SPT.L", "SRP.L", "SSTL.L", "STJ.L", "SWJ.L", "TCG.L",
    "TED.L", "TEM.L", "TLW.L", "TPK.L", "TRMR.L", "TRN.L", "TUNE.L",
    "UTG.L", "VCT.L", "VEC.L", "VNET.L", "VTY.L", "WIZZ.L", "WKF.L",
    "WOSG.L", "WPS.L", "XAR.L", "YGEN.L",
]

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT SCREENING THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_FILTERS = {
    # Valuation multiples
    "max_pe":            25.0,   # Trailing P/E ratio (exclude expensive or loss-making)
    "min_pe":             5.0,   # Exclude distressed / zero-earnings
    "max_pb":             4.0,   # Price-to-Book
    "max_ev_ebitda":     15.0,   # EV/EBITDA
    "min_div_yield":      0.015, # Minimum dividend yield (1.5%)

    # Momentum signals (price returns)
    "min_momentum_3m":    0.0,   # 3-month return > 0 (positive medium-term momentum)
    "min_momentum_6m":    0.0,   # 6-month return > 0
    "min_momentum_12m":   0.0,   # 12-month return > 0 (annual trend filter)

    # Quality / risk filters
    "min_market_cap_m":  500,    # Minimum market cap £m (avoid micro-caps)
    "min_volume":      50000,    # Minimum average daily volume
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_fundamentals(ticker: str) -> dict:
    """Fetch key fundamental and price data for a single ticker."""
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        market_cap = info.get("marketCap", None)
        market_cap_m = market_cap / 1_000_000 if market_cap else None

        return {
            "ticker":        ticker,
            "name":          info.get("longName", ticker),
            "sector":        info.get("sector", "N/A"),
            "industry":      info.get("industry", "N/A"),
            "price":         info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap_m":  round(market_cap_m, 0) if market_cap_m else None,
            "pe_ratio":      info.get("trailingPE"),
            "pb_ratio":      info.get("priceToBook"),
            "ev_ebitda":     info.get("enterpriseToEbitda"),
            "div_yield":     info.get("dividendYield"),
            "avg_volume":    info.get("averageVolume"),
            "52w_high":      info.get("fiftyTwoWeekHigh"),
            "52w_low":       info.get("fiftyTwoWeekLow"),
            "beta":          info.get("beta"),
        }
    except Exception:
        return {"ticker": ticker, "name": ticker, "sector": "N/A"}


def fetch_momentum(ticker: str, today: datetime.date) -> dict:
    """Fetch 3m, 6m, 12m price momentum for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        start = today - datetime.timedelta(days=370)
        hist  = stock.history(start=start.strftime("%Y-%m-%d"),
                              end=today.strftime("%Y-%m-%d"),
                              auto_adjust=True)
        if hist.empty or len(hist) < 60:
            return {}

        close = hist["Close"]
        p_now = close.iloc[-1]

        def ret(days):
            idx = max(0, len(close) - days)
            return (p_now / close.iloc[idx] - 1) if close.iloc[idx] > 0 else None

        return {
            "momentum_3m":  round(ret(63),  4) if ret(63)  is not None else None,
            "momentum_6m":  round(ret(126), 4) if ret(126) is not None else None,
            "momentum_12m": round(ret(252), 4) if ret(252) is not None else None,
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# SCREENER ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_screener(
    tickers: list = None,
    filters: dict = None,
    verbose: bool = True,
    **filter_overrides
) -> pd.DataFrame:
    """
    Run the FTSE 350 equity screener.

    Parameters
    ----------
    tickers : list, optional
        List of Yahoo Finance tickers to screen. Defaults to FTSE_350_TICKERS.
    filters : dict, optional
        Full filter dictionary. Defaults to DEFAULT_FILTERS.
    verbose : bool
        Print progress to console.
    **filter_overrides
        Override individual filter values, e.g. max_pe=20, min_momentum_3m=0.05

    Returns
    -------
    pd.DataFrame
        Screened results, sorted by P/E ratio ascending.

    Examples
    --------
    # Default screen
    results = run_screener()

    # Custom thresholds
    results = run_screener(max_pe=15, min_div_yield=0.03, min_momentum_3m=0.05)

    # Value screen (low P/E, low P/B, positive momentum)
    results = run_screener(max_pe=12, max_pb=1.5, min_momentum_12m=0.0)

    # Income screen (high dividend yield)
    results = run_screener(min_div_yield=0.04, max_pe=20)
    """
    if tickers is None:
        tickers = FTSE_350_TICKERS
    if filters is None:
        filters = DEFAULT_FILTERS.copy()
    filters.update(filter_overrides)

    today = datetime.date.today()

    if verbose:
        print("=" * 65)
        print("  FTSE 350 EQUITY SCREENER")
        print(f"  Run date : {today.strftime('%d %B %Y')}")
        print(f"  Universe : {len(tickers)} tickers")
        print("=" * 65)
        print("\nActive filters:")
        for k, v in filters.items():
            print(f"  {k:<22} {v}")
        print()

    # ── Fetch data ───────────────────────────────────────────────────────────
    records = []
    for i, ticker in enumerate(tickers, 1):
        if verbose and i % 20 == 0:
            print(f"  Fetching {i}/{len(tickers)}...")
        fundamentals = fetch_fundamentals(ticker)
        momentum     = fetch_momentum(ticker, today)
        records.append({**fundamentals, **momentum})

    df = pd.DataFrame(records)

    # ── Apply filters ────────────────────────────────────────────────────────
    mask = pd.Series([True] * len(df))

    def safe_filter(col, op, threshold):
        nonlocal mask
        if col not in df.columns:
            return
        col_data = pd.to_numeric(df[col], errors="coerce")
        if op == "<=":
            mask &= col_data.fillna(np.inf) <= threshold
        elif op == ">=":
            mask &= col_data.fillna(-np.inf) >= threshold

    safe_filter("pe_ratio",     "<=", filters.get("max_pe",            np.inf))
    safe_filter("pe_ratio",     ">=", filters.get("min_pe",            -np.inf))
    safe_filter("pb_ratio",     "<=", filters.get("max_pb",            np.inf))
    safe_filter("ev_ebitda",    "<=", filters.get("max_ev_ebitda",     np.inf))
    safe_filter("div_yield",    ">=", filters.get("min_div_yield",     -np.inf))
    safe_filter("momentum_3m",  ">=", filters.get("min_momentum_3m",   -np.inf))
    safe_filter("momentum_6m",  ">=", filters.get("min_momentum_6m",   -np.inf))
    safe_filter("momentum_12m", ">=", filters.get("min_momentum_12m",  -np.inf))
    safe_filter("market_cap_m", ">=", filters.get("min_market_cap_m",  -np.inf))
    safe_filter("avg_volume",   ">=", filters.get("min_volume",        -np.inf))

    results = df[mask].copy()

    # ── Format output ────────────────────────────────────────────────────────
    pct_cols = ["div_yield", "momentum_3m", "momentum_6m", "momentum_12m"]
    for col in pct_cols:
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce")
            results[col] = results[col].map(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
            )

    for col in ["pe_ratio", "pb_ratio", "ev_ebitda", "beta"]:
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce").round(1)

    if "price" in results.columns:
        results["price"] = pd.to_numeric(results["price"], errors="coerce").round(2)
    if "market_cap_m" in results.columns:
        results["market_cap_m"] = pd.to_numeric(
            results["market_cap_m"], errors="coerce"
        ).map(lambda x: f"£{x:,.0f}m" if pd.notna(x) else "N/A")

    display_cols = [
        "ticker", "name", "sector", "price", "market_cap_m",
        "pe_ratio", "pb_ratio", "ev_ebitda", "div_yield",
        "momentum_3m", "momentum_6m", "momentum_12m",
        "52w_low", "52w_high", "beta",
    ]
    display_cols = [c for c in display_cols if c in results.columns]
    results = results[display_cols].sort_values("pe_ratio")

    if verbose:
        print(f"\n{'='*65}")
        print(f"  RESULTS: {len(results)} stocks passed all filters")
        print(f"{'='*65}\n")
        if len(results) > 0:
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 200)
            pd.set_option("display.max_rows", 100)
            print(results.to_string(index=False))
        else:
            print("  No stocks passed all filters. Try relaxing thresholds.")
        print()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PRESET SCREENS
# ─────────────────────────────────────────────────────────────────────────────

def value_screen() -> pd.DataFrame:
    """Low P/E, low P/B, positive 12-month momentum."""
    print("\n>>> VALUE SCREEN: Low multiples + positive annual momentum\n")
    return run_screener(max_pe=14, max_pb=2.0, min_momentum_12m=0.0,
                        min_market_cap_m=500)


def income_screen() -> pd.DataFrame:
    """High dividend yield with reasonable valuation."""
    print("\n>>> INCOME SCREEN: High yield + positive momentum\n")
    return run_screener(min_div_yield=0.04, max_pe=20,
                        min_momentum_6m=0.0, min_market_cap_m=500)


def momentum_screen() -> pd.DataFrame:
    """Strong price momentum across all time horizons."""
    print("\n>>> MOMENTUM SCREEN: Positive 3m, 6m, 12m momentum\n")
    return run_screener(min_momentum_3m=0.05, min_momentum_6m=0.08,
                        min_momentum_12m=0.10, max_pe=30, min_market_cap_m=500)


def quality_value_screen() -> pd.DataFrame:
    """Balanced quality-value screen: fair multiple, dividend, positive trend."""
    print("\n>>> QUALITY-VALUE SCREEN\n")
    return run_screener(max_pe=18, max_pb=3.0, max_ev_ebitda=12,
                        min_div_yield=0.02, min_momentum_6m=0.0,
                        min_market_cap_m=1000)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("""
  ┌─────────────────────────────────────────────────────┐
  │         FTSE 350 EQUITY SCREENER — Roman Falla      │
  │                                                     │
  │  Screens by valuation multiples & momentum signals  │
  │  Universe: FTSE 350 (LSE-listed, .L tickers)        │
  └─────────────────────────────────────────────────────┘
    """)

    print("Select a screen:")
    print("  1. Default screen (P/E ≤ 25, P/B ≤ 4, positive momentum)")
    print("  2. Value screen   (P/E ≤ 14, P/B ≤ 2, 12m momentum > 0)")
    print("  3. Income screen  (Yield ≥ 4%, P/E ≤ 20)")
    print("  4. Momentum screen (3m > 5%, 6m > 8%, 12m > 10%)")
    print("  5. Quality-value  (P/E ≤ 18, EV/EBITDA ≤ 12, yield ≥ 2%)")
    print()

    choice = input("Enter choice (1-5) or press Enter for default: ").strip()

    if choice == "2":
        results = value_screen()
    elif choice == "3":
        results = income_screen()
    elif choice == "4":
        results = momentum_screen()
    elif choice == "5":
        results = quality_value_screen()
    else:
        results = run_screener()

    # Save results
    if len(results) > 0:
        out_file = f"ftse350_screen_results_{datetime.date.today()}.csv"
        # Strip percentage signs before saving to CSV
        save_df = results.copy()
        results.to_csv(out_file, index=False)
        print(f"\nResults saved to: {out_file}")
    else:
        print("\nNo results to save. Try relaxing filter thresholds.")
