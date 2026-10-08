"""
FTSE 350 Equity Screener
========================
Screens a sample of FTSE 350 stocks using valuation, dividend,
market-cap and price-momentum signals.

Author: Roman Falla
GitHub: github.com/romanfalla343-jpg

Dependencies:
    pip install yfinance pandas numpy

Usage:
    python ftse350_equity_screener.py

Or import and customise thresholds:
    from ftse350_equity_screener import run_screener
    results = run_screener(max_pe=20, min_momentum_3m=0.05)
"""

import datetime
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# FTSE 350 TICKERS
# A hand-picked sample of FTSE 350 constituents.
# This is not the full FTSE 350 universe.
# Yahoo Finance uses the .L suffix for London Stock Exchange securities.
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
    "MNDI.L", "MNG.L", "NG.L", "NWG.L", "NXT.L", "OCDO.L",
    "PCT.L", "PHNX.L", "PRU.L", "PSH.L", "PSN.L", "PSON.L", "RB.L",
    "REL.L", "RIO.L", "RKT.L", "RMV.L", "RR.L", "RS1.L", "RTO.L",
    "SBRY.L", "SDR.L", "SGE.L", "SGRO.L", "SHEL.L", "SKG.L", "SMDS.L",
    "SMIN.L", "SMT.L", "SN.L", "SPX.L", "SSE.L", "STAN.L", "SVT.L",
    "TSCO.L", "TW.L", "ULVR.L", "UU.L", "VOD.L", "WEIR.L", "WPP.L",
    "WTB.L",

    # FTSE 250 selection
    "ABG.L", "ACSO.L", "AGK.L", "AML.L", "BNZL.L", "BOWL.L", "BTG.L",
    "CAL.L", "CASH.L", "CBG.L", "CLG.L", "CMC.L", "COB.L", "CTEC.L",
    "CVS.L", "DARK.L", "DNLM.L", "DTY.L", "ECM.L", "EMG.L", "ENTR.L",
    "ESNT.L", "FDM.L", "FGP.L", "FLTK.L", "FSV.L", "GNC.L", "GPOR.L",
    "GRI.L", "GRG.L", "GTLS.L", "HAT.L", "HBR.L", "HFD.L", "HMSO.L",
    "HUW.L", "HWDN.L", "IHG.L", "IMI.L", "INCH.L", "ITV.L", "JET2.L",
    "JUP.L", "KIE.L", "LAD.L", "LIO.L", "LRE.L", "MCS.L", "MERI.L",
    "MGAM.L", "MKS.L", "MNKS.L", "MPI.L", "MTM.L", "MTO.L", "MWE.L",
    "NCT.L", "NETW.L", "NXRT.L", "OSB.L", "OXB.L", "PAG.L", "PCA.L",
    "PETS.L", "PFC.L", "PMVD.L", "PNN.L", "POLR.L", "PZC.L", "QQ.L",
    "RDW.L", "RGD.L", "RHI.L", "RHIM.L", "RNK.L", "SAFE.L", "SCT.L",
    "SHI.L", "SLA.L", "SLP.L", "SNN.L", "SPT.L", "SRP.L", "SSTL.L",
    "STJ.L", "SWJ.L", "TCG.L", "TED.L", "TEM.L", "TLW.L", "TPK.L",
    "TRMR.L", "TRN.L", "TUNE.L", "UTG.L", "VCT.L", "VEC.L", "VNET.L",
    "VTY.L", "WIZZ.L", "WKF.L", "WOSG.L", "WPS.L", "XAR.L", "YGEN.L",
]


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT SCREENING THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_FILTERS = {
    "max_pe": 25.0,
    "min_pe": 5.0,
    "max_pb": 4.0,
    "max_ev_ebitda": 15.0,
    "min_div_yield": 0.015,

    "min_momentum_3m": 0.0,
    "min_momentum_6m": 0.0,
    "min_momentum_12m": 0.0,

    "min_market_cap_m": 500,
    "min_volume": 50000,
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _norm_yield(y):
    """Return dividend yield as a decimal fraction."""
    if y is None or pd.isna(y):
        return None

    try:
        y = float(y)
    except (TypeError, ValueError):
        return None

    # yfinance can return either 0.045 or 4.5 for a 4.5% yield.
    if y > 1:
        y /= 100

    return y


def fetch_fundamentals(ticker: str) -> dict:
    """Fetch fundamental and market data for one ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        market_cap = info.get("marketCap")
        market_cap_m = (
            market_cap / 1_000_000
            if market_cap is not None
            else None
        )

        return {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "price": (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
            ),
            "market_cap_m": (
                round(market_cap_m, 0)
                if market_cap_m is not None
                else None
            ),
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "div_yield": _norm_yield(
                info.get("dividendYield")
            ),
            "avg_volume": info.get("averageVolume"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "beta": info.get("beta"),
        }

    except Exception as exc:
        return {
            "ticker": ticker,
            "name": ticker,
            "sector": "N/A",
            "_fetch_error": str(exc),
        }


def fetch_momentum(ticker: str, today: datetime.date) -> dict:
    """
    Calculate 3m, 6m and 12m price momentum.

    Uses approximately 63, 126 and 252 trading-day windows.
    Returns None where insufficient price history exists.
    """
    try:
        stock = yf.Ticker(ticker)

        start = today - datetime.timedelta(days=370)

        hist = stock.history(
            start=start.strftime("%Y-%m-%d"),
            end=today.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )

        if hist.empty:
            return {}

        close = hist["Close"].dropna()

        def ret(days):
            # Need at least days + 1 observations to calculate
            # a genuine return from t-days to t.
            if len(close) <= days:
                return None

            past_price = close.iloc[-1 - days]
            current_price = close.iloc[-1]

            if past_price <= 0:
                return None

            return current_price / past_price - 1

        return {
            "momentum_3m": ret(63),
            "momentum_6m": ret(126),
            "momentum_12m": ret(252),
        }

    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# SCREENING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_screener(
    tickers: list = None,
    filters: dict = None,
    verbose: bool = True,
    **filter_overrides,
) -> pd.DataFrame:
    """
    Run the equity screener.

    Parameters
    ----------
    tickers : list, optional
        Yahoo Finance tickers to screen.
        Defaults to FTSE_350_TICKERS.

    filters : dict, optional
        Complete filter dictionary.
        Defaults to DEFAULT_FILTERS.

    verbose : bool
        Print progress and results.

    **filter_overrides
        Override individual filters.

    Returns
    -------
    pd.DataFrame
        Numeric screening results sorted by P/E.
    """

    if tickers is None:
        tickers = FTSE_350_TICKERS

    if filters is None:
        active_filters = DEFAULT_FILTERS.copy()
    else:
        active_filters = filters.copy()

    active_filters.update(filter_overrides)

    today = datetime.date.today()

    if verbose:
        print("=" * 70)
        print("  FTSE 350 EQUITY SCREENER")
        print(f"  Run date : {today.strftime('%d %B %Y')}")
        print(f"  Universe : {len(tickers)} tickers")
        print("=" * 70)

        print("\nActive filters:")
        for key, value in active_filters.items():
            print(f"  {key:<22} {value}")

        print()

    # ── Fetch data ───────────────────────────────────────────────────────────

    records = []
    failed_tickers = []

    for i, ticker in enumerate(tickers, 1):

        if verbose and i % 20 == 0:
            print(f"  Fetching {i}/{len(tickers)}...")

        fundamentals = fetch_fundamentals(ticker)
        momentum = fetch_momentum(ticker, today)

        if fundamentals.get("_fetch_error"):
            failed_tickers.append(ticker)

        records.append({
            **fundamentals,
            **momentum,
        })

    df = pd.DataFrame(records)

    if "_fetch_error" in df.columns:
        df = df.drop(columns=["_fetch_error"])

    # ── Apply filters ────────────────────────────────────────────────────────

    mask = pd.Series(True, index=df.index)

    def safe_filter(col, op, threshold):
        """
        Apply a filter while allowing unavailable data to remain in
        the universe rather than automatically failing the stock.
        """
        nonlocal mask

        if col not in df.columns:
            return

        col_data = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        available = col_data.notna()

        if op == "<=":
            mask &= (~available) | (col_data <= threshold)

        elif op == ">=":
            mask &= (~available) | (col_data >= threshold)

    safe_filter(
        "pe_ratio",
        "<=",
        active_filters.get("max_pe", np.inf),
    )

    safe_filter(
        "pe_ratio",
        ">=",
        active_filters.get("min_pe", -np.inf),
    )

    # Prevent negative P/B and EV/EBITDA values from passing
    # valuation screens.
    safe_filter("pb_ratio", ">=", 0)
    safe_filter(
        "pb_ratio",
        "<=",
        active_filters.get("max_pb", np.inf),
    )

    safe_filter("ev_ebitda", ">=", 0)
    safe_filter(
        "ev_ebitda",
        "<=",
        active_filters.get("max_ev_ebitda", np.inf),
    )

    safe_filter(
        "div_yield",
        ">=",
        active_filters.get("min_div_yield", -np.inf),
    )

    safe_filter(
        "momentum_3m",
        ">=",
        active_filters.get("min_momentum_3m", -np.inf),
    )

    safe_filter(
        "momentum_6m",
        ">=",
        active_filters.get("min_momentum_6m", -np.inf),
    )

    safe_filter(
        "momentum_12m",
        ">=",
        active_filters.get("min_momentum_12m", -np.inf),
    )

    safe_filter(
        "market_cap_m",
        ">=",
        active_filters.get("min_market_cap_m", -np.inf),
    )

    safe_filter(
        "avg_volume",
        ">=",
        active_filters.get("min_volume", -np.inf),
    )

    results = df.loc[mask].copy()

    # ── Keep returned data numeric ──────────────────────────────────────────

    for col in [
        "pe_ratio",
        "pb_ratio",
        "ev_ebitda",
        "beta",
    ]:
        if col in results.columns:
            results[col] = pd.to_numeric(
                results[col],
                errors="coerce",
            ).round(1)

    if "price" in results.columns:
        results["price"] = pd.to_numeric(
            results["price"],
            errors="coerce",
        ).round(2)

    if "market_cap_m" in results.columns:
        results["market_cap_m"] = pd.to_numeric(
            results["market_cap_m"],
            errors="coerce",
        ).round(0)

    if "div_yield" in results.columns:
        results["div_yield"] = pd.to_numeric(
            results["div_yield"],
            errors="coerce",
        ).round(4)

    for col in [
        "momentum_3m",
        "momentum_6m",
        "momentum_12m",
    ]:
        if col in results.columns:
            results[col] = pd.to_numeric(
                results[col],
                errors="coerce",
            ).round(4)

    # ── Select and sort output columns ──────────────────────────────────────

    display_cols = [
        "ticker",
        "name",
        "sector",
        "price",
        "market_cap_m",
        "pe_ratio",
        "pb_ratio",
        "ev_ebitda",
        "div_yield",
        "momentum_3m",
        "momentum_6m",
        "momentum_12m",
        "52w_low",
        "52w_high",
        "beta",
    ]

    display_cols = [
        col for col in display_cols
        if col in results.columns
    ]

    results = results[display_cols]

    if "pe_ratio" in results.columns:
        results = results.sort_values(
            "pe_ratio",
            na_position="last",
        )

    # ── Console output ──────────────────────────────────────────────────────

    if verbose:
        print(f"\n{'=' * 70}")
        print(
            f"  RESULTS: {len(results)} stocks passed all filters"
        )
        print(f"{'=' * 70}\n")

        if failed_tickers:
            print(
                f"  Warning: {len(failed_tickers)} tickers "
                "failed to return fundamental data."
            )
            print(
                "  Failed tickers: "
                + ", ".join(failed_tickers)
            )
            print()

        if len(results) > 0:
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 220)
            pd.set_option("display.max_rows", 100)

            print(results.to_string(index=False))

        else:
            print(
                "  No stocks passed all filters. "
                "Try relaxing thresholds."
            )

        print()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PRESET SCREENS
# ─────────────────────────────────────────────────────────────────────────────

def value_screen() -> pd.DataFrame:
    """Low P/E, low P/B and positive 12-month momentum."""

    print(
        "\n>>> VALUE SCREEN: "
        "Low multiples + positive annual momentum\n"
    )

    filters = {
        "max_pe": 14,
        "min_pe": 5,
        "max_pb": 2.0,
        "min_momentum_12m": 0.0,
        "min_market_cap_m": 500,
    }

    return run_screener(filters=filters)


def income_screen() -> pd.DataFrame:
    """High dividend yield with reasonable valuation."""

    print(
        "\n>>> INCOME SCREEN: "
        "High yield + positive 6-month momentum\n"
    )

    filters = {
        "max_pe": 20,
        "min_pe": 5,
        "min_div_yield": 0.04,
        "min_momentum_6m": 0.0,
        "min_market_cap_m": 500,
    }

    return run_screener(filters=filters)


def momentum_screen() -> pd.DataFrame:
    """Strong price momentum across all time horizons."""

    print(
        "\n>>> MOMENTUM SCREEN: "
        "Positive 3m, 6m and 12m momentum\n"
    )

    filters = {
        "min_momentum_3m": 0.05,
        "min_momentum_6m": 0.08,
        "min_momentum_12m": 0.10,
        "min_market_cap_m": 500,
    }

    return run_screener(filters=filters)


def quality_value_screen() -> pd.DataFrame:
    """Balanced quality-value screen."""

    print("\n>>> QUALITY-VALUE SCREEN\n")

    filters = {
        "max_pe": 18,
        "min_pe": 5,
        "max_pb": 3.0,
        "max_ev_ebitda": 12,
        "min_div_yield": 0.02,
        "min_momentum_6m": 0.0,
        "min_market_cap_m": 1000,
    }

    return run_screener(filters=filters)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(
        """
  ┌─────────────────────────────────────────────────────┐
  │         FTSE 350 EQUITY SCREENER — Roman Falla      │
  │                                                     │
  │  Screens by valuation multiples & momentum signals  │
  │  Universe: FTSE 350 sample (LSE-listed, .L)         │
  └─────────────────────────────────────────────────────┘
        """
    )

    print("Select a screen:")
    print(
        "  1. Default screen "
        "(P/E ≤ 25, P/B ≤ 4, positive momentum)"
    )
    print(
        "  2. Value screen   "
        "(P/E ≤ 14, P/B ≤ 2, 12m momentum > 0)"
    )
    print(
        "  3. Income screen  "
        "(Yield ≥ 4%, P/E ≤ 20, 6m momentum > 0)"
    )
    print(
        "  4. Momentum screen "
        "(3m > 5%, 6m > 8%, 12m > 10%)"
    )
    print(
        "  5. Quality-value  "
        "(P/E ≤ 18, EV/EBITDA ≤ 12, yield ≥ 2%)"
    )
    print()

    choice = input(
        "Enter choice (1-5) or press Enter for default: "
    ).strip()

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

    # ── Save results ─────────────────────────────────────────────────────────

    if len(results) > 0:

        out_file = (
            f"ftse350_screen_results_"
            f"{datetime.date.today()}.csv"
        )

        results.to_csv(
            out_file,
            index=False,
        )

        print(
            f"\nResults saved to: {out_file}"
        )

    else:

        print(
            "\nNo results to save. "
            "Try relaxing filter thresholds."
        )
