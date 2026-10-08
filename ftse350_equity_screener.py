"""
FTSE 350 Equity Screener
========================
Screens FTSE 350 stocks using valuation, dividend,
market-cap and price-momentum signals.

Author: Roman Falla
"""

import datetime
import warnings
import logging

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


# ─────────────────────────────────────────────────────────────────────────────
# FTSE 350 SAMPLE UNIVERSE
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

    # FTSE 250 sample
    "ABG.L", "ACSO.L", "AGK.L", "AML.L", "BNZL.L", "BOWL.L", "BTG.L",
    "CAL.L", "CASH.L", "CBG.L", "CLG.L", "CMC.L", "COB.L", "CTEC.L",
    "CVS.L", "DARK.L", "DNLM.L", "DTY.L", "ECM.L", "EMG.L", "ENTR.L",
    "ESNT.L", "FDM.L", "FGP.L", "FSV.L", "GNC.L", "GPOR.L",
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
# DEFAULT FILTERS
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
# DIVIDEND YIELD
# ─────────────────────────────────────────────────────────────────────────────

def get_dividend_yield(stock, price):
    """
    Calculate trailing 12-month dividend yield directly from
    Yahoo Finance dividend history.

    Formula:

        Dividend Yield =
            trailing 12-month dividends / current share price

    This avoids relying on Yahoo's dividendYield fields,
    which can have inconsistent units for LSE securities.
    """

    if price is None:
        return None

    try:
        price = float(price)

        if price <= 0:
            return None

        # Pull approximately 15 months so that we have enough
        # history around the 12-month boundary.
        dividends = stock.dividends

        if dividends is None or dividends.empty:
            return 0.0

        dividends = dividends.dropna()

        if dividends.empty:
            return 0.0

        # Make timezone handling safe.
        try:
            if dividends.index.tz is not None:
                dividends.index = dividends.index.tz_localize(None)
        except Exception:
            pass

        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)

        trailing_dividends = dividends[
            dividends.index >= cutoff
        ]

        if trailing_dividends.empty:
            return 0.0

        annual_dividends = trailing_dividends.sum()

        if annual_dividends < 0:
            return None

        dividend_yield = (
            annual_dividends / price
        )

        # Reject clearly impossible values.
        if dividend_yield > 0.25:
            return None

        return float(dividend_yield)

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FUNDAMENTALS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_fundamentals(ticker):

    try:

        stock = yf.Ticker(ticker)

        info = stock.info

        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
        )

        market_cap = info.get("marketCap")

        if price is None and market_cap is None:

            return {
                "ticker": ticker,
                "_fetch_error": True,
            }

        market_cap_m = (
            market_cap / 1_000_000
            if market_cap is not None
            else None
        )

        # Calculate dividend yield directly from
        # actual dividend payments.
        div_yield = get_dividend_yield(
            stock,
            price
        )

        return {

            "ticker": ticker,

            "name": info.get(
                "longName",
                ticker
            ),

            "sector": info.get(
                "sector",
                "N/A"
            ),

            "price": price,

            "market_cap_m": market_cap_m,

            "pe_ratio": info.get(
                "trailingPE"
            ),

            "pb_ratio": info.get(
                "priceToBook"
            ),

            "ev_ebitda": info.get(
                "enterpriseToEbitda"
            ),

            "div_yield": div_yield,

            "avg_volume": info.get(
                "averageVolume"
            ),

            "52w_high": info.get(
                "fiftyTwoWeekHigh"
            ),

            "52w_low": info.get(
                "fiftyTwoWeekLow"
            ),

            "beta": info.get(
                "beta"
            ),
        }

    except Exception:

        return {
            "ticker": ticker,
            "_fetch_error": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# MOMENTUM
# ─────────────────────────────────────────────────────────────────────────────

def fetch_momentum(ticker, today):

    try:

        stock = yf.Ticker(ticker)

        start = today - datetime.timedelta(
            days=370
        )

        hist = stock.history(
            start=start.strftime("%Y-%m-%d"),
            end=today.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )

        if hist.empty:
            return {}

        close = hist["Close"].dropna()

        def calculate_return(days):

            if len(close) <= days:
                return None

            past_price = close.iloc[-1 - days]
            current_price = close.iloc[-1]

            if past_price <= 0:
                return None

            return (
                current_price / past_price
            ) - 1

        return {

            "momentum_3m":
                calculate_return(63),

            "momentum_6m":
                calculate_return(126),

            "momentum_12m":
                calculate_return(252),

        }

    except Exception:

        return {}


# ─────────────────────────────────────────────────────────────────────────────
# SCREENING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_screener(
    tickers=None,
    filters=None,
    verbose=True,
    **filter_overrides,
):

    if tickers is None:
        tickers = FTSE_350_TICKERS

    if filters is None:
        active_filters = DEFAULT_FILTERS.copy()
    else:
        active_filters = filters.copy()

    active_filters.update(
        filter_overrides
    )

    today = datetime.date.today()

    if verbose:

        print("=" * 70)

        print(
            "  FTSE 350 EQUITY SCREENER"
        )

        print(
            f"  Run date : "
            f"{today.strftime('%d %B %Y')}"
        )

        print(
            f"  Universe : "
            f"{len(tickers)} tickers"
        )

        print("=" * 70)

        print("\nActive filters:")

        for key, value in active_filters.items():

            print(
                f"  {key:<22} {value}"
            )

        print()

    records = []

    failed_tickers = []

    # ── Fetch data ───────────────────────────────────────────────────────────

    for i, ticker in enumerate(
        tickers,
        1
    ):

        if verbose and i % 20 == 0:

            print(
                f"  Fetching "
                f"{i}/{len(tickers)}..."
            )

        fundamentals = fetch_fundamentals(
            ticker
        )

        if fundamentals.get(
            "_fetch_error"
        ):

            failed_tickers.append(
                ticker
            )

            continue

        momentum = fetch_momentum(
            ticker,
            today
        )

        records.append(
            {
                **fundamentals,
                **momentum
            }
        )

    if not records:

        print(
            "\nNo usable market data."
        )

        return pd.DataFrame()

    df = pd.DataFrame(
        records
    )

    # ── Filtering ───────────────────────────────────────────────────────────

    mask = pd.Series(
        True,
        index=df.index
    )

    def strict_filter(
        column,
        operator,
        threshold,
    ):

        nonlocal mask

        if column not in df.columns:
            return

        data = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if operator == "<=":

            mask &= (
                data.notna()
                & (data <= threshold)
            )

        elif operator == ">=":

            mask &= (
                data.notna()
                & (data >= threshold)
            )

    # P/E
    if "min_pe" in active_filters:

        strict_filter(
            "pe_ratio",
            ">=",
            active_filters["min_pe"]
        )

    if "max_pe" in active_filters:

        strict_filter(
            "pe_ratio",
            "<=",
            active_filters["max_pe"]
        )

    # P/B
    if "max_pb" in active_filters:

        strict_filter(
            "pb_ratio",
            ">=",
            0
        )

        strict_filter(
            "pb_ratio",
            "<=",
            active_filters["max_pb"]
        )

    # EV/EBITDA
    if "max_ev_ebitda" in active_filters:

        strict_filter(
            "ev_ebitda",
            ">=",
            0
        )

        strict_filter(
            "ev_ebitda",
            "<=",
            active_filters["max_ev_ebitda"]
        )

    # Dividend yield
    if "min_div_yield" in active_filters:

        strict_filter(
            "div_yield",
            ">=",
            active_filters["min_div_yield"]
        )

    # Momentum
    if "min_momentum_3m" in active_filters:

        strict_filter(
            "momentum_3m",
            ">=",
            active_filters["min_momentum_3m"]
        )

    if "min_momentum_6m" in active_filters:

        strict_filter(
            "momentum_6m",
            ">=",
            active_filters["min_momentum_6m"]
        )

    if "min_momentum_12m" in active_filters:

        strict_filter(
            "momentum_12m",
            ">=",
            active_filters["min_momentum_12m"]
        )

    # Market cap
    if "min_market_cap_m" in active_filters:

        strict_filter(
            "market_cap_m",
            ">=",
            active_filters["min_market_cap_m"]
        )

    # Volume
    if "min_volume" in active_filters:

        strict_filter(
            "avg_volume",
            ">=",
            active_filters["min_volume"]
        )

    results = df.loc[
        mask
    ].copy()

    # ── Formatting ───────────────────────────────────────────────────────────

    numeric_columns = [
        "pe_ratio",
        "pb_ratio",
        "ev_ebitda",
        "beta",
        "price",
        "market_cap_m",
        "div_yield",
        "momentum_3m",
        "momentum_6m",
        "momentum_12m",
    ]

    for column in numeric_columns:

        if column in results.columns:

            results[column] = pd.to_numeric(
                results[column],
                errors="coerce"
            )

    for column in [
        "pe_ratio",
        "pb_ratio",
        "ev_ebitda",
        "beta",
    ]:

        if column in results.columns:

            results[column] = results[
                column
            ].round(1)

    if "price" in results.columns:

        results["price"] = results[
            "price"
        ].round(2)

    if "market_cap_m" in results.columns:

        results["market_cap_m"] = results[
            "market_cap_m"
        ].round(0)

    if "div_yield" in results.columns:

        results["div_yield"] = results[
            "div_yield"
        ].round(4)

    for column in [
        "momentum_3m",
        "momentum_6m",
        "momentum_12m",
    ]:

        if column in results.columns:

            results[column] = results[
                column
            ].round(4)

    # ── Output columns ──────────────────────────────────────────────────────

    display_columns = [

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

    display_columns = [
        column
        for column in display_columns
        if column in results.columns
    ]

    results = results[
        display_columns
    ]

    # Sort by strongest 12-month momentum
    if "momentum_12m" in results.columns:

        results = results.sort_values(
            "momentum_12m",
            ascending=False,
            na_position="last"
        )

    # ── Print results ───────────────────────────────────────────────────────

    if verbose:

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"  RESULTS: "
            f"{len(results)} stocks "
            f"passed all filters"
        )

        print(
            f"{'=' * 70}"
        )

        print(
            f"\n  Successfully retrieved: "
            f"{len(df)}"
        )

        print(
            f"  Unavailable/invalid:    "
            f"{len(failed_tickers)}"
        )

        if failed_tickers:

            print(
                "\n  Excluded tickers:"
            )

            print(
                "  "
                + ", ".join(
                    failed_tickers
                )
            )

        print()

        if len(results) > 0:

            pd.set_option(
                "display.max_columns",
                None
            )

            pd.set_option(
                "display.width",
                220
            )

            pd.set_option(
                "display.max_rows",
                100
            )

            print(
                results.to_string(
                    index=False
                )
            )

        else:

            print(
                "  No stocks passed "
                "all filters."
            )

        print()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PRESET SCREENS
# ─────────────────────────────────────────────────────────────────────────────

def value_screen():

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

    return run_screener(
        filters=filters
    )


def income_screen():

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

    return run_screener(
        filters=filters
    )


def momentum_screen():

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

    return run_screener(
        filters=filters
    )


def quality_value_screen():

    print(
        "\n>>> QUALITY-VALUE SCREEN\n"
    )

    filters = {

        "max_pe": 18,
        "min_pe": 5,
        "max_pb": 3.0,
        "max_ev_ebitda": 12,
        "min_div_yield": 0.02,
        "min_momentum_6m": 0.0,
        "min_market_cap_m": 1000,

    }

    return run_screener(
        filters=filters
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROGRAM
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
        "  2. Value screen "
        "(P/E ≤ 14, P/B ≤ 2, 12m momentum > 0)"
    )

    print(
        "  3. Income screen "
        "(Yield ≥ 4%, P/E ≤ 20, 6m momentum > 0)"
    )

    print(
        "  4. Momentum screen "
        "(3m > 5%, 6m > 8%, 12m > 10%)"
    )

    print(
        "  5. Quality-value "
        "(P/E ≤ 18, EV/EBITDA ≤ 12, yield ≥ 2%)"
    )

    print()

    choice = input(
        "Enter choice (1-5) "
        "or press Enter for default: "
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
            "ftse350_screen_results_"
            f"{datetime.date.today()}.csv"
        )

        results.to_csv(
            out_file,
            index=False
        )

        print(
            f"Results saved to: "
            f"{out_file}"
        )

    else:

        print(
            "No results to save."
        )
