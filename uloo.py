import math

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

# ============================================================
# SMARTVEST - PORTFOLIO MANAGEMENT DASHBOARD
# Everything is contained in this single app.py file.
# ============================================================

st.set_page_config(
    page_title="SmartVest | Investment Dashboard",
    page_icon="📈",
    layout="wide",
)

INITIAL_INVESTMENT = 1_00_00_000
GS_WEIGHT = 10.0
GS_RATE = 6.03
GS_INVESTMENT = INITIAL_INVESTMENT * GS_WEIGHT / 100
EQUITY_BUDGET = INITIAL_INVESTMENT - GS_INVESTMENT
GS_NAME = "Government Security GS 2029"

BENCHMARKS = {
    "Nifty 50": "^NSEI",
    "Nifty 500": "^CRSLDX",
}

COMPANIES = {
    "SBI": "SBIN.NS",
    "Trent": "TRENT.NS",
    "Larsen & Toubro": "LT.NS",
    "Cholamandalam Finance": "CHOLAFIN.NS",
    "Persistent Systems": "PERSISTENT.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Torrent Pharma": "TORNTPHARM.NS",
    "Solar Industries": "SOLARINDS.NS",
    "Welspun Corp": "WELCORP.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Marico": "MARICO.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Hindustan Aeronautics (HAL)": "HAL.NS",
    "Garden Reach Shipbuilders (GRSE)": "GRSE.NS",
    "Waaree Renewable Technologies": "WAAREERTL.NS",
}

SECTORS = {
    "SBI": "Banking",
    "Trent": "Retail",
    "Larsen & Toubro": "Infrastructure / Capital Goods",
    "Cholamandalam Finance": "Financial Services",
    "Persistent Systems": "IT Services",
    "Sun Pharma": "Pharmaceuticals",
    "Torrent Pharma": "Pharmaceuticals",
    "Solar Industries": "Industrials / Explosives",
    "Welspun Corp": "Industrials / Pipes",
    "Reliance Industries": "Diversified",
    "Marico": "Consumer Goods",
    "ICICI Bank": "Banking",
    "Hindustan Aeronautics (HAL)": "Defence / Capital Goods",
    "Garden Reach Shipbuilders (GRSE)": "Defence / Shipbuilding",
    "Waaree Renewable Technologies": "Renewable Energy",
}


# ============================================================
# CACHED DATA FUNCTIONS
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def load_prices(start_date, end_date):
    """Download adjusted daily stock prices."""

    tickers = list(COMPANIES.values())

    try:
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date + timedelta(days=1),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            level0 = data.columns.get_level_values(0)
            level1 = data.columns.get_level_values(1)

            if "Close" in level0:
                result = data["Close"].copy()
            elif "Close" in level1:
                result = data.xs("Close", axis=1, level=1).copy()
            else:
                return pd.DataFrame()
        else:
            if "Close" not in data.columns:
                return pd.DataFrame()

            result = data[["Close"]].copy()
            result.columns = [tickers[0]]

        result = result.rename(
            columns={
                ticker: company
                for company, ticker in COMPANIES.items()
            }
        )

        result.index = pd.to_datetime(result.index)
        result = result[~result.index.duplicated(keep="last")]
        result = result.sort_index()
        result = result.dropna(axis=1, how="all")

        return result

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def load_benchmark(ticker, start_date, end_date):
    """Download a benchmark index."""

    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date + timedelta(days=1),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if data.empty:
            return pd.Series(dtype=float)

        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.get_level_values(0):
                series = data["Close"]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
            else:
                return pd.Series(dtype=float)
        else:
            if "Close" not in data.columns:
                return pd.Series(dtype=float)
            series = data["Close"]

        series = series.dropna()
        series.index = pd.to_datetime(series.index)

        return series.sort_index()

    except Exception:
        return pd.Series(dtype=float)


# ------------------------------------------------------------
# FUNDAMENTALS: Yahoo `.info` first, calculated fallback second
# ------------------------------------------------------------

def _row(df, names):
    """First non-empty statement row (newest value first) matching a name."""

    if df is None or not hasattr(df, "empty") or df.empty:
        return None

    for name in names:
        if name in df.index:
            series = df.loc[name].dropna()
            if not series.empty:
                return series

    return None


def _ok(x):
    return (
        isinstance(x, (int, float, np.integer, np.floating))
        and not isinstance(x, bool)
        and np.isfinite(x)
    )


def _first(series):
    return float(series.iloc[0]) if series is not None else None


def _growth(series):
    """Latest vs previous period growth (decimal)."""

    if series is None or len(series) < 2:
        return None

    latest, previous = float(series.iloc[0]), float(series.iloc[1])

    if previous == 0:
        return None

    return latest / abs(previous) - 1


@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamentals(ticker):
    """
    Returns (info, values, sources).

    Each value comes from Yahoo's `.info` when available. Otherwise it is
    calculated from the annual financial statements. `sources` records
    whether each value is 'Yahoo', 'Calculated' or 'Unavailable'.
    """

    t = yf.Ticker(ticker)

    try:
        info = t.info or {}
    except Exception:
        info = {}

    def grab(attr):
        try:
            return getattr(t, attr)
        except Exception:
            return None

    inc = grab("financials")
    bs = grab("balance_sheet")
    cf = grab("cashflow")

    # ---- price and share count ----
    price = info.get("currentPrice")

    if not _ok(price):
        try:
            price = float(t.fast_info["last_price"])
        except Exception:
            price = None

    shares = info.get("sharesOutstanding")

    if not _ok(shares):
        s = _row(bs, ["Ordinary Shares Number", "Share Issued"])
        shares = _first(s)

    # ---- statement rows ----
    ni_s = _row(inc, ["Net Income Common Stockholders", "Net Income"])
    rev_s = _row(inc, ["Total Revenue", "Operating Revenue"])
    gp_s = _row(inc, ["Gross Profit"])
    op_s = _row(inc, ["Operating Income", "EBIT"])
    eq_s = _row(bs, ["Stockholders Equity", "Common Stock Equity"])
    as_s = _row(bs, ["Total Assets"])
    debt_s = _row(bs, ["Total Debt"])
    cash_s = _row(
        bs,
        [
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents",
        ],
    )
    ca_s = _row(bs, ["Current Assets"])
    cl_s = _row(bs, ["Current Liabilities"])
    fcf_s = _row(cf, ["Free Cash Flow"])

    if fcf_s is None:
        ocf = _row(
            cf,
            [
                "Operating Cash Flow",
                "Cash Flow From Continuing Operating Activities",
            ],
        )
        capex = _row(cf, ["Capital Expenditure"])

        if ocf is not None and capex is not None:
            # Yahoo reports capital expenditure as a negative number.
            fcf_s = (ocf + capex).dropna()
            if fcf_s.empty:
                fcf_s = None

    ni, rev = _first(ni_s), _first(rev_s)
    gp, op = _first(gp_s), _first(op_s)
    eq, assets = _first(eq_s), _first(as_s)
    debt, cash = _first(debt_s), _first(cash_s)
    ca, cl, fcf = _first(ca_s), _first(cl_s), _first(fcf_s)

    # ---- trailing 12-month dividend yield from actual payouts ----
    div_yield = None

    try:
        d = t.dividends

        if d is not None and len(d) and _ok(price) and price > 0:
            d = d.copy()
            d.index = pd.to_datetime(d.index).tz_localize(None)

            cutoff = pd.Timestamp.today() - pd.Timedelta(days=365)
            last12 = float(d[d.index >= cutoff].sum())

            div_yield = last12 / price
    except Exception:
        div_yield = None

    mcap = info.get("marketCap")

    if not _ok(mcap) and _ok(price) and _ok(shares):
        mcap = price * shares

    eps = info.get("trailingEps")

    if not _ok(eps) and ni is not None and _ok(shares) and shares > 0:
        eps = ni / shares

    computed = {
        "price": price,
        "mcap": mcap,
        "pe": price / eps if _ok(price) and _ok(eps) and eps > 0 else None,
        "fpe": None,  # needs analyst estimates; Yahoo only
        "pb": mcap / eq if _ok(mcap) and eq and eq > 0 else None,
        "eps": eps,
        "beta": None,  # calculated on the page from price data
        "roe": ni / eq if ni is not None and eq else None,
        "roa": ni / assets if ni is not None and assets else None,
        "pm": ni / rev if ni is not None and rev else None,
        "om": op / rev if op is not None and rev else None,
        "gm": gp / rev if gp is not None and rev else None,
        "rg": _growth(rev_s),
        "eg": _growth(ni_s),
        "rev": rev,
        "ni": ni,
        "debt": debt,
        "cash": cash,
        # Yahoo reports debt-to-equity as a percentage.
        "de": debt / eq * 100 if debt is not None and eq and eq > 0 else None,
        "cr": ca / cl if ca and cl else None,
        "fcf": fcf,
    }

    info_keys = {
        "price": "currentPrice",
        "mcap": "marketCap",
        "pe": "trailingPE",
        "fpe": "forwardPE",
        "pb": "priceToBook",
        "eps": "trailingEps",
        "beta": "beta",
        "roe": "returnOnEquity",
        "roa": "returnOnAssets",
        "pm": "profitMargins",
        "om": "operatingMargins",
        "gm": "grossMargins",
        "rg": "revenueGrowth",
        "eg": "earningsGrowth",
        "rev": "totalRevenue",
        "ni": "netIncomeToCommon",
        "debt": "totalDebt",
        "cash": "totalCash",
        "de": "debtToEquity",
        "cr": "currentRatio",
        "fcf": "freeCashflow",
    }

    values, sources = {}, {}

    for key, calc in computed.items():
        yahoo_value = info.get(info_keys[key])

        if _ok(yahoo_value):
            values[key], sources[key] = float(yahoo_value), "Yahoo"
        elif _ok(calc):
            values[key], sources[key] = float(calc), "Calculated"
        else:
            values[key], sources[key] = None, "Unavailable"

    # Dividend yield: always calculated (Yahoo's units have changed
    # between versions).
    if _ok(div_yield):
        values["div"], sources["div"] = float(div_yield), "Calculated"
    else:
        values["div"], sources["div"] = None, "Unavailable"

    return info, values, sources


# ------------------------------------------------------------
# FORMATTERS
# ------------------------------------------------------------

def money(value):
    """Format Indian rupees."""

    if value is None or pd.isna(value):
        return "N/A"

    return f"₹{value:,.2f}"


def percentage(value):
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.2f}%"


def safe_number(value, decimals=3):
    if value is None or pd.isna(value) or not np.isfinite(value):
        return "N/A"

    return f"{value:.{decimals}f}"


# ------------------------------------------------------------
# PORTFOLIO CONSTRUCTION
# ------------------------------------------------------------

def build_portfolio(prices, weights):
    """
    Build an illustrative portfolio.

    Each equity allocation is invested at its first available
    adjusted closing price in the selected period.

    The government security is held at cost in the value history.
    Its actual market return is not available in this application.
    """

    if prices.empty:
        return None

    available = [
        name for name in weights
        if name in prices.columns and prices[name].notna().any()
    ]

    missing = [
        name for name in weights
        if weights[name] > 0 and name not in available
    ]

    index = prices.index
    stock_values = pd.DataFrame(index=index)

    allocated_amounts = {
        name: INITIAL_INVESTMENT * weight / 100
        for name, weight in weights.items()
    }

    starting_values = {}
    ending_values = {}
    period_returns = {}

    for name in weights:
        amount = allocated_amounts[name]

        if name not in available:
            stock_values[name] = amount
            starting_values[name] = amount
            ending_values[name] = amount
            period_returns[name] = np.nan
            continue

        series = prices[name].dropna().sort_index()

        if series.empty or series.iloc[0] <= 0:
            stock_values[name] = amount
            starting_values[name] = amount
            ending_values[name] = amount
            period_returns[name] = np.nan
            continue

        first_price = float(series.iloc[0])
        shares = amount / first_price

        filled_prices = prices[name].reindex(index).ffill()
        value_series = filled_prices * shares

        first_valid_date = series.index[0]
        value_series.loc[value_series.index < first_valid_date] = amount
        value_series = value_series.fillna(amount)

        stock_values[name] = value_series
        starting_values[name] = amount
        ending_values[name] = float(value_series.iloc[-1])

        period_returns[name] = (
            float(series.iloc[-1]) / first_price - 1
        ) * 100

    equity_values = stock_values.sum(axis=1)

    # The government security is held at cost in this history.
    portfolio_values = equity_values + GS_INVESTMENT

    portfolio_returns = (
        portfolio_values
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    daily_std = portfolio_returns.std(ddof=1)
    daily_variance = portfolio_returns.var(ddof=1)

    annual_volatility = (
        daily_std * np.sqrt(252) * 100
        if pd.notna(daily_std)
        else np.nan
    )

    equity_value = float(equity_values.iloc[-1])
    portfolio_value = float(portfolio_values.iloc[-1])

    total_pnl = portfolio_value - INITIAL_INVESTMENT
    total_pnl_pct = total_pnl / INITIAL_INVESTMENT * 100

    daily_gain = (
        float(portfolio_values.iloc[-1] - portfolio_values.iloc[-2])
        if len(portfolio_values) >= 2
        else np.nan
    )

    history = pd.DataFrame({
        "Equity Value (₹)": equity_values,
        "Government Security (₹)": GS_INVESTMENT,
        "Estimated Portfolio Value (₹)": portfolio_values,
    })

    return {
        "available": available,
        "missing": missing,
        "stock_values": stock_values,
        "starting_values": starting_values,
        "ending_values": ending_values,
        "period_returns": period_returns,
        "equity_values": equity_values,
        "portfolio_values": portfolio_values,
        "portfolio_returns": portfolio_returns,
        "history": history,
        "equity_value": equity_value,
        "portfolio_value": portfolio_value,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "daily_gain": daily_gain,
        "daily_std": daily_std,
        "daily_variance": daily_variance,
        "annual_volatility": annual_volatility,
    }


# ------------------------------------------------------------
# CAPM
# ------------------------------------------------------------

def _clean_index(series):
    """Return a copy with a timezone-naive, midnight-normalised index."""

    s = series.copy()
    idx = pd.to_datetime(s.index)

    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)

    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")]

    return s.sort_index()


def get_benchmark_returns(benchmark_name, start, end):
    """Daily decimal returns of the chosen market benchmark."""

    prices_series = load_benchmark(BENCHMARKS[benchmark_name], start, end)

    if prices_series.empty:
        return pd.Series(dtype=float)

    returns = (
        _clean_index(prices_series)
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    return returns


def fit_capm(asset_returns, market_returns, rf_annual, min_obs=30):
    """
    Estimate CAPM by OLS on daily excess returns:

        R_i - R_f = alpha + beta * (R_m - R_f) + error

    Inputs are daily decimal returns. rf_annual is a percentage.
    """

    asset = _clean_index(asset_returns.dropna())
    market = _clean_index(market_returns.dropna())

    combined = pd.concat(
        [asset.rename("Asset"), market.rename("Market")],
        axis=1,
        join="inner",
    ).replace([np.inf, -np.inf], np.nan).dropna()

    n = len(combined)

    if n < min_obs:
        raise ValueError(
            f"Only {n} overlapping daily observations were found; at least "
            f"{min_obs} are needed. Select a wider date range."
        )

    daily_rf = (1 + rf_annual / 100) ** (1 / 252) - 1

    y = combined["Asset"] - daily_rf
    x = combined["Market"] - daily_rf

    x_var = x.var(ddof=1)

    if not np.isfinite(x_var) or x_var <= 1e-14:
        raise ValueError("Market return variance is zero; cannot estimate beta.")

    beta = float(np.cov(x, y, ddof=1)[0, 1] / x_var)
    alpha = float(y.mean() - beta * x.mean())

    residuals = y - (alpha + beta * x)
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    sigma2 = ss_res / (n - 2)
    sxx = float(np.sum((x - x.mean()) ** 2))

    se_beta = math.sqrt(sigma2 / sxx) if sxx > 0 else np.nan
    se_alpha = (
        math.sqrt(sigma2 * (1 / n + x.mean() ** 2 / sxx))
        if sxx > 0
        else np.nan
    )

    t_beta = beta / se_beta if se_beta and se_beta > 0 else np.nan
    t_alpha = alpha / se_alpha if se_alpha and se_alpha > 0 else np.nan

    def p_value(t_stat):
        # Normal approximation; adequate for n >= 30.
        if t_stat is None or not np.isfinite(t_stat):
            return np.nan
        return math.erfc(abs(t_stat) / math.sqrt(2))

    return {
        "alpha_daily": alpha,
        "alpha_annual_pct": ((1 + alpha) ** 252 - 1) * 100,
        "beta": beta,
        "r_squared": r_squared,
        "se_beta": se_beta,
        "t_alpha": t_alpha,
        "t_beta": t_beta,
        "p_alpha": p_value(t_alpha),
        "p_beta": p_value(t_beta),
        "observations": n,
        "daily_rf": daily_rf,
        "combined": combined,
    }


def annualised_return(returns):
    """Compounded annualised return (decimal) from daily returns."""

    r = returns.dropna()

    if len(r) < 2:
        return np.nan

    growth = float((1 + r).prod())

    if growth <= 0:
        return np.nan

    return growth ** (252 / len(r)) - 1


# ============================================================
# SIDEBAR: NAVIGATION, DATE RANGE AND WEIGHTS
# ============================================================

st.sidebar.title("📈 SmartVest")
st.sidebar.caption("Investment & Portfolio Dashboard")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Security Analysis",
        "Fundamentals",
        "Portfolio Analysis",
        "CAPM Analysis",
        "Summary",
        "Investment Simulator",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("Analysis Period")

today = date.today()
default_start = today - timedelta(days=3 * 365)

analysis_start = st.sidebar.date_input(
    "Start date",
    value=default_start,
    max_value=today,
)

analysis_end = st.sidebar.date_input(
    "End date",
    value=today,
    min_value=analysis_start,
    max_value=today,
)

if analysis_start > analysis_end:
    st.sidebar.error("Start date must be before the end date.")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("Equity Allocation")

st.sidebar.caption(
    "Equities must total 90% of the ₹1 crore portfolio. "
    "The Government Security allocation remains fixed at 10%."
)

weights = {}

for company in COMPANIES:
    weights[company] = 6.0

with st.sidebar.expander(
    "Edit individual stock weights",
    expanded=False,
):
    for company in COMPANIES:
        weights[company] = st.number_input(
            company,
            min_value=0.0,
            max_value=90.0,
            value=6.0,
            step=0.5,
            key=f"weight_{company}",
        )

total_equity_weight = sum(weights.values())
valid_allocation = abs(total_equity_weight - 90.0) < 0.01

st.sidebar.metric(
    "Total Equity Weight",
    f"{total_equity_weight:.1f}%",
)

st.sidebar.metric("Government Security", "10.0%")

if not valid_allocation:
    st.sidebar.error(
        f"Equity weights total {total_equity_weight:.1f}%. "
        "Adjust them to exactly 90% for a fully allocated portfolio."
    )

st.sidebar.caption(
    "Historical price data comes from Yahoo Finance. "
    "Figures are estimates, not live broker holdings."
)


# ============================================================
# LOAD SELECTED PERIOD DATA
# ============================================================

with st.spinner("Loading historical market data..."):
    prices = load_prices(analysis_start, analysis_end)

if prices.empty:
    st.error(
        "No historical price data could be loaded. "
        "Check your internet connection or select another date range."
    )
    st.stop()

portfolio = build_portfolio(prices, weights)

if portfolio is None:
    st.error("Unable to construct the portfolio from the available data.")
    st.stop()

st.sidebar.divider()

if st.sidebar.button("Refresh market data"):
    load_prices.clear()
    load_benchmark.clear()
    get_fundamentals.clear()
    st.rerun()


def target_returns(name):
    """Daily decimal returns for a company or the estimated portfolio."""

    if name == "Estimated portfolio":
        return (
            portfolio["portfolio_values"]
            .pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

    if name in prices.columns:
        return (
            prices[name]
            .dropna()
            .pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

    return pd.Series(dtype=float)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.title("SmartVest Portfolio Overview")

    st.caption(
        f"Analysis period: {analysis_start:%d %b %Y} – "
        f"{analysis_end:%d %b %Y}"
    )

    st.subheader("Portfolio Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Initial Investment", money(INITIAL_INVESTMENT))

    c2.metric(
        "Estimated Portfolio Value",
        money(portfolio["portfolio_value"]),
    )

    c3.metric(
        "Overall Profit / Loss",
        money(portfolio["total_pnl"]),
        f'{portfolio["total_pnl_pct"]:.2f}%',
    )

    daily_text = (
        money(portfolio["daily_gain"])
        if pd.notna(portfolio["daily_gain"])
        else "N/A"
    )

    c4.metric("Latest Daily Change", daily_text)

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric("Equity Value", money(portfolio["equity_value"]))

    c2.metric(
        "Government Security Allocation",
        money(GS_INVESTMENT),
    )

    c3.metric(
        "Annualised Portfolio Volatility",
        percentage(portfolio["annual_volatility"]),
    )

    if portfolio["missing"]:
        st.warning(
            "No usable historical price series was found for: "
            + ", ".join(portfolio["missing"])
            + ". Their allocations are retained at cost, so their actual "
            "performance is not reflected."
        )

    st.subheader("Portfolio Value Over Time")

    st.line_chart(
        portfolio["history"],
        y=[
            "Equity Value (₹)",
            "Government Security (₹)",
            "Estimated Portfolio Value (₹)",
        ],
    )

    st.subheader("Asset Allocation")

    allocation_df = pd.DataFrame({
        "Asset": ["Equities", GS_NAME],
        "Allocation (₹)": [EQUITY_BUDGET, GS_INVESTMENT],
        "Weight (%)": [90.0, 10.0],
    })

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            allocation_df,
            names="Asset",
            values="Allocation (₹)",
            title="Equity vs Government Security",
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(
            allocation_df,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Top Gainers and Losers")

    performance_rows = []

    for company in COMPANIES:
        ret = portfolio["period_returns"].get(company, np.nan)

        performance_rows.append({
            "Company": company,
            "Weight (%)": weights[company],
            "Allocation (₹)": INITIAL_INVESTMENT * weights[company] / 100,
            "Selected Period Return (%)": ret,
            "Estimated Current Value (₹)": portfolio[
                "ending_values"
            ].get(company, np.nan),
        })

    performance_df = pd.DataFrame(performance_rows)

    valid_performance = performance_df.dropna(
        subset=["Selected Period Return (%)"]
    )

    if not valid_performance.empty:
        top_col, bottom_col = st.columns(2)

        with top_col:
            st.markdown("**Top Gainers**")

            gainers = valid_performance.nlargest(
                5, "Selected Period Return (%)"
            )

            st.dataframe(
                gainers[["Company", "Selected Period Return (%)"]],
                use_container_width=True,
                hide_index=True,
            )

        with bottom_col:
            st.markdown("**Top Losers**")

            losers = valid_performance.nsmallest(
                5, "Selected Period Return (%)"
            )

            st.dataframe(
                losers[["Company", "Selected Period Return (%)"]],
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Stock Allocation")

    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        performance_df,
        x="Company",
        y="Weight (%)",
        title="Individual Stock Weights",
    )

    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sector Exposure")

    sector_rows = []

    for company, weight in weights.items():
        sector_rows.append({
            "Sector": SECTORS.get(company, "Other"),
            "Weight (%)": weight,
            "Allocation (₹)": INITIAL_INVESTMENT * weight / 100,
        })

    sector_df = pd.DataFrame(sector_rows)

    sector_df = sector_df.groupby(
        "Sector", as_index=False
    ).sum(numeric_only=True)

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(
            sector_df.sort_values("Weight (%)", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        fig = px.bar(
            sector_df,
            x="Sector",
            y="Weight (%)",
            title="Sector Concentration",
        )

        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Sector labels are broad categories assigned in the code. "
        "They are not a live classification from an exchange."
    )

    st.subheader("Portfolio vs Market Benchmark")

    benchmark_choice = st.selectbox(
        "Choose benchmark",
        list(BENCHMARKS.keys()),
        key="overview_benchmark",
    )

    benchmark_prices = load_benchmark(
        BENCHMARKS[benchmark_choice],
        analysis_start,
        analysis_end,
    )

    if not benchmark_prices.empty:

        portfolio_index = (
            portfolio["portfolio_values"]
            / portfolio["portfolio_values"].iloc[0]
            * 100
        )

        benchmark_index = (
            benchmark_prices / benchmark_prices.iloc[0] * 100
        )

        comparison = pd.concat(
            [
                _clean_index(portfolio_index).rename("SmartVest Portfolio"),
                _clean_index(benchmark_index).rename(benchmark_choice),
            ],
            axis=1,
            join="inner",
        ).dropna()

        if not comparison.empty:
            st.line_chart(comparison)
        else:
            st.info(
                "There are no overlapping dates for the benchmark comparison."
            )

    else:
        st.info(
            "Benchmark data could not be retrieved for this date range."
        )

    st.subheader("Government Security: GS 2029")

    gs_df = pd.DataFrame({
        "Instrument": [GS_NAME],
        "Portfolio Allocation (₹)": [GS_INVESTMENT],
        "Portfolio Weight (%)": [GS_WEIGHT],
        "Reference Rate (%)": [GS_RATE],
        "Illustrative Annual Amount (₹)": [
            GS_INVESTMENT * GS_RATE / 100
        ],
    })

    st.dataframe(
        gs_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The 6.03% rate is a reference assumption. The illustrative annual "
        "amount is not added to portfolio profit or loss. Actual bond value "
        "and cash flows require verified prices, coupon terms, payment "
        "schedule and maturity details."
    )


# ============================================================
# SECURITY ANALYSIS
# ============================================================

elif page == "Security Analysis":

    st.title("Security Analysis")
    st.caption("Historical price, return and risk analysis")

    # --------------------------------------------------------
    # SUMMARY TABLE FOR ALL SECURITIES
    # --------------------------------------------------------

    st.subheader("Security Summary")

    summary_rows = []

    for company in COMPANIES:
        row = {
            "Security": company,
            "Beginning Close": np.nan,
            "Ending Close": np.nan,
            "Return": np.nan,
            "Standard Deviation": np.nan,
            "Variance": np.nan,
        }

        if company in prices.columns:
            s = prices[company].dropna().sort_index()

            if len(s) >= 2 and s.iloc[0] > 0:
                r = (
                    s.pct_change(fill_method=None)
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )

                row["Beginning Close"] = float(s.iloc[0])
                row["Ending Close"] = float(s.iloc[-1])
                row["Return"] = (s.iloc[-1] / s.iloc[0] - 1) * 100
                row["Standard Deviation"] = r.std(ddof=1) * 100
                row["Variance"] = r.var(ddof=1)

        summary_rows.append(row)

    summary_numeric = pd.DataFrame(summary_rows)

    summary_display = pd.DataFrame({
        "Security": summary_numeric["Security"],
        "Beginning Close": summary_numeric["Beginning Close"].map(money),
        "Ending Close": summary_numeric["Ending Close"].map(money),
        "Return": summary_numeric["Return"].map(percentage),
        "Standard Deviation": summary_numeric["Standard Deviation"].map(percentage),
        "Variance": summary_numeric["Variance"].map(lambda v: safe_number(v, 6)),
    })

    st.dataframe(
        summary_display,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Period: {analysis_start:%d %b %Y} – {analysis_end:%d %b %Y}. "
        "Beginning and ending close are the first and last adjusted closing "
        "prices available in the period. Return is ending ÷ beginning − 1. "
        "Standard deviation and variance are calculated from daily returns "
        "(not annualised)."
    )

    st.download_button(
        "Download Summary Table",
        data=summary_numeric.to_csv(index=False).encode("utf-8"),
        file_name="security_summary.csv",
        mime="text/csv",
    )

    # --------------------------------------------------------
    # SINGLE SECURITY DETAIL
    # --------------------------------------------------------

    st.divider()
    st.subheader("Single Security Detail")

    selected_company = st.selectbox(
        "Select Company",
        list(COMPANIES.keys()),
    )

    if selected_company not in prices.columns:
        st.warning(
            "No price data is available for this company "
            "in the selected date range."
        )
        st.stop()

    company_prices = prices[selected_company].dropna().sort_index()

    if len(company_prices) < 2:
        st.warning(
            "At least two historical price observations are needed. "
            "Select a wider date range."
        )
        st.stop()

    daily_returns = company_prices.pct_change(fill_method=None)
    daily_changes = company_prices.diff()

    period_return = (
        company_prices.iloc[-1] / company_prices.iloc[0] - 1
    ) * 100

    valid_returns = daily_returns.dropna()

    daily_std = valid_returns.std(ddof=1)
    daily_variance = valid_returns.var(ddof=1)

    annualised_volatility = daily_std * np.sqrt(252) * 100

    latest_daily_return = (
        daily_returns.iloc[-1] * 100
        if pd.notna(daily_returns.iloc[-1])
        else np.nan
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Latest Adjusted Close", money(company_prices.iloc[-1]))
    col2.metric("Selected Period Return", percentage(period_return))
    col3.metric("Latest Daily Return", percentage(latest_daily_return))
    col4.metric("Annualised Volatility", percentage(annualised_volatility))

    st.subheader("Historical Price Chart")

    st.line_chart(company_prices.rename("Adjusted Closing Price (₹)"))

    st.subheader("Return Chart")

    cumulative_returns = (
        company_prices / company_prices.iloc[0] - 1
    ) * 100

    st.line_chart(cumulative_returns.rename("Cumulative Return (%)"))

    analysis_table = pd.DataFrame({
        "Date": company_prices.index,
        "Adjusted Close (₹)": company_prices.values,
        "Daily Change (₹)": daily_changes.values,
        "Daily Return (%)": daily_returns.values * 100,
        "Cumulative Return (%)": cumulative_returns.values,
        "Rolling Annualised Volatility (%)": (
            daily_returns.rolling(30).std() * np.sqrt(252) * 100
        ).values,
        "Rolling Daily Return Variance": (
            daily_returns.rolling(30).var()
        ).values,
    })

    analysis_table["Date"] = pd.to_datetime(
        analysis_table["Date"]
    ).dt.strftime("%d-%m-%Y")

    numeric_columns = analysis_table.select_dtypes(include="number").columns

    analysis_table[numeric_columns] = (
        analysis_table[numeric_columns].round(5)
    )

    analysis_table = analysis_table.iloc[::-1].reset_index(drop=True)

    with st.expander(f"Daily data tables: {selected_company}"):

        price_tab, return_tab, risk_tab = st.tabs([
            "Price Analysis",
            "Return Analysis",
            "Risk Analysis",
        ])

        with price_tab:
            st.dataframe(
                analysis_table[
                    ["Date", "Adjusted Close (₹)", "Daily Change (₹)"]
                ],
                use_container_width=True,
                hide_index=True,
            )

        with return_tab:
            st.dataframe(
                analysis_table[
                    ["Date", "Daily Return (%)", "Cumulative Return (%)"]
                ],
                use_container_width=True,
                hide_index=True,
            )

        with risk_tab:
            risk1, risk2, risk3 = st.columns(3)

            risk1.metric("Daily Standard Deviation", safe_number(daily_std, 6))
            risk2.metric("Daily Return Variance", safe_number(daily_variance, 8))
            risk3.metric("Annualised Volatility", percentage(annualised_volatility))

            negative_returns = valid_returns[valid_returns < 0]

            downside_deviation = (
                negative_returns.std(ddof=1) * np.sqrt(252) * 100
                if len(negative_returns) >= 2
                else np.nan
            )

            st.metric("Downside Deviation", percentage(downside_deviation))

            st.write(
                "Rolling risk measures use 30 trading observations. "
                "Annualised volatility is daily standard deviation "
                "multiplied by √252."
            )

            st.dataframe(
                analysis_table[
                    [
                        "Date",
                        "Rolling Annualised Volatility (%)",
                        "Rolling Daily Return Variance",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        csv_data = analysis_table.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Complete Price, Return & Risk Data",
            data=csv_data,
            file_name=f"{selected_company.replace(' ', '_')}_analysis.csv",
            mime="text/csv",
        )


# ============================================================
# FUNDAMENTALS
# ============================================================

elif page == "Fundamentals":

    st.title("Fundamental Analysis")
    st.caption(
        "Company financial indicators. Missing Yahoo values are calculated "
        "from the annual financial statements where possible."
    )

    selected_company = st.selectbox(
        "Select Company",
        list(COMPANIES.keys()),
    )

    ticker = COMPANIES[selected_company]

    with st.spinner("Retrieving company fundamentals..."):
        info, fv, src = get_fundamentals(ticker)

    if not info and all(v is None for v in fv.values()):
        st.warning(
            "Fundamental data could not be retrieved for this company. "
            "Try refreshing market data or selecting another company."
        )
        st.stop()

    # Beta: use Yahoo if given, otherwise calculate against the Nifty 50
    # from the price data already loaded for the selected period.
    if fv.get("beta") is None and selected_company in prices.columns:
        try:
            market_ret = get_benchmark_returns(
                "Nifty 50", analysis_start, analysis_end
            )
            stock_ret = target_returns(selected_company)

            aligned = pd.concat(
                [
                    _clean_index(stock_ret).rename("S"),
                    _clean_index(market_ret).rename("M"),
                ],
                axis=1,
                join="inner",
            ).dropna()

            if len(aligned) >= 30 and aligned["M"].var(ddof=1) > 1e-14:
                fv["beta"] = float(
                    aligned["S"].cov(aligned["M"]) / aligned["M"].var(ddof=1)
                )
                src["beta"] = "Calculated (vs Nifty 50, selected period)"
        except Exception:
            pass

    def fmt(key, kind):
        v = fv.get(key)

        if v is None:
            return "Unavailable"

        if kind == "num":
            return f"{v:,.2f}"
        if kind == "pct":
            return f"{v * 100:.2f}%"
        if kind == "big":
            return f"₹{v:,.0f}"

        return str(v)

    def fundamentals_table(rows):
        return pd.DataFrame({
            "Indicator": [r[0] for r in rows],
            "Value": [fmt(r[1], r[2]) for r in rows],
            "Source": [src.get(r[1], "Unavailable") for r in rows],
        })

    st.subheader(info.get("longName", selected_company))

    st.write(
        f"**Sector:** {info.get('sector', 'Unavailable')}  \n"
        f"**Industry:** {info.get('industry', 'Unavailable')}  \n"
        f"**Exchange ticker:** {ticker}"
    )

    st.subheader("Market and Valuation")

    st.dataframe(
        fundamentals_table([
            ("Current / Latest Market Price (₹)", "price", "num"),
            ("Market Capitalisation", "mcap", "big"),
            ("Trailing P/E", "pe", "num"),
            ("Forward P/E", "fpe", "num"),
            ("Price-to-Book Ratio", "pb", "num"),
            ("Earnings Per Share (EPS)", "eps", "num"),
            ("Dividend Yield (trailing 12 months)", "div", "pct"),
            ("Beta", "beta", "num"),
        ]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Profitability and Growth")

    st.dataframe(
        fundamentals_table([
            ("Return on Equity (ROE)", "roe", "pct"),
            ("Return on Assets (ROA)", "roa", "pct"),
            ("Profit Margin", "pm", "pct"),
            ("Operating Margin", "om", "pct"),
            ("Gross Margin", "gm", "pct"),
            ("Revenue Growth (YoY)", "rg", "pct"),
            ("Earnings Growth (YoY)", "eg", "pct"),
            ("Total Revenue", "rev", "big"),
            ("Net Income", "ni", "big"),
        ]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Debt and Liquidity")

    st.dataframe(
        fundamentals_table([
            ("Total Debt", "debt", "big"),
            ("Total Cash", "cash", "big"),
            ("Debt-to-Equity (%)", "de", "num"),
            ("Current Ratio", "cr", "num"),
            ("Free Cash Flow", "fcf", "big"),
        ]),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "**Source** shows where each number came from: 'Yahoo' is Yahoo "
        "Finance's summary data; 'Calculated' is derived here from the "
        "latest annual financial statements (and may differ slightly from "
        "TTM figures shown elsewhere); 'Unavailable' means neither source "
        "had enough data. Forward P/E needs analyst estimates and cannot be "
        "calculated from statements. For banks and NBFCs (SBI, ICICI Bank, "
        "Cholamandalam), gross margin, current ratio and free cash flow are "
        "not meaningful, and debt ratios should be read differently."
    )


# ============================================================
# PORTFOLIO ANALYSIS
# ============================================================

elif page == "Portfolio Analysis":

    st.title("Portfolio Analysis")
    st.caption("Allocation, covariance, return and risk metrics")

    control1, control2 = st.columns(2)

    with control1:
        benchmark_name = st.selectbox(
            "Benchmark for performance metrics",
            list(BENCHMARKS.keys()),
            key="portfolio_benchmark",
        )

    with control2:
        rf_annual = st.number_input(
            "Assumed annual risk-free rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=6.03,
            step=0.25,
            key="portfolio_rf_annual",
            help=(
                "Assumption for risk-adjusted performance metrics. "
                "This is not a verified market yield."
            ),
        )

    benchmark_prices = load_benchmark(
        BENCHMARKS[benchmark_name],
        analysis_start,
        analysis_end,
    )

    portfolio_value = portfolio["portfolio_values"].dropna().sort_index()

    portfolio_returns = (
        portfolio_value
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    daily_rf = (1 + rf_annual / 100) ** (1 / 252) - 1

    excess_returns = (portfolio_returns - daily_rf).dropna()

    daily_return_std = portfolio_returns.std(ddof=1)

    if (
        len(excess_returns) >= 2
        and pd.notna(daily_return_std)
        and daily_return_std > 1e-12
    ):
        sharpe_ratio = (
            excess_returns.mean() / daily_return_std
        ) * np.sqrt(252)
    else:
        sharpe_ratio = np.nan

    st.subheader("Portfolio Summary")

    s1, s2, s3, s4, s5 = st.columns(5)

    s1.metric("Portfolio Value", money(portfolio["portfolio_value"]))

    s2.metric(
        "Total P&L",
        money(portfolio["total_pnl"]),
        percentage(portfolio["total_pnl_pct"]),
    )

    s3.metric(
        "Annualised Volatility",
        percentage(portfolio["annual_volatility"]),
    )

    s4.metric(
        "Daily Variance",
        safe_number(portfolio["daily_variance"], 8),
    )

    s5.metric(
        "Sharpe Ratio",
        safe_number(sharpe_ratio, 3),
        help=(
            "Annualised average daily excess return divided by daily "
            "portfolio return standard deviation. A longer history is "
            "recommended."
        ),
    )

    if len(portfolio_returns) < 30:
        st.info(
            f"There are {len(portfolio_returns)} valid daily portfolio "
            "returns. Risk metrics based on short periods can be unstable; "
            "consider selecting a wider date range."
        )

    if pd.isna(sharpe_ratio):
        st.warning(
            "Sharpe Ratio is unavailable because there are too few valid "
            "returns or the portfolio return volatility is zero."
        )

    if portfolio["missing"]:
        st.warning(
            "Missing price data for: "
            + ", ".join(portfolio["missing"])
            + ". Risk and return calculations may not fully represent "
            "these securities."
        )

    st.divider()

    st.subheader("Security Weights")

    weight_df = pd.DataFrame({
        "Security": list(weights.keys()),
        "Weight (%)": list(weights.values()),
        "Initial Allocation (₹)": [
            INITIAL_INVESTMENT * weight / 100
            for weight in weights.values()
        ],
        "Period Return (%)": [
            portfolio["period_returns"].get(name, np.nan)
            for name in weights
        ],
        "Estimated Current Value (₹)": [
            portfolio["ending_values"].get(name, np.nan)
            for name in weights
        ],
    })

    weight_df = pd.concat(
        [
            weight_df,
            pd.DataFrame([{
                "Security": GS_NAME,
                "Weight (%)": GS_WEIGHT,
                "Initial Allocation (₹)": GS_INVESTMENT,
                "Period Return (%)": np.nan,
                "Estimated Current Value (₹)": GS_INVESTMENT,
            }]),
        ],
        ignore_index=True,
    )

    st.dataframe(weight_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        weight_df,
        x="Security",
        y="Weight (%)",
        title="Portfolio Security Weights",
    )

    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Portfolio Return Chart")

    if not portfolio_value.empty and portfolio_value.iloc[0] != 0:
        rebased_portfolio = (
            portfolio_value / portfolio_value.iloc[0] - 1
        ) * 100

        st.line_chart(rebased_portfolio.rename("Portfolio Return (%)"))
    else:
        st.info("Portfolio return chart is unavailable for this period.")

    st.subheader("Covariance Matrix of Daily Stock Returns")

    available = [
        name for name in portfolio["available"]
        if name in prices.columns
    ]

    if len(available) >= 2:

        return_matrix = (
            prices[available]
            .sort_index()
            .pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
        )

        covariance_matrix = return_matrix.cov()
        correlation_matrix = return_matrix.corr()

        cov_tab, corr_tab = st.tabs([
            "Covariance Matrix",
            "Correlation Matrix",
        ])

        with cov_tab:
            st.dataframe(
                covariance_matrix.round(8),
                use_container_width=True,
            )

            st.download_button(
                "Download Covariance Matrix",
                data=covariance_matrix.to_csv().encode("utf-8"),
                file_name="covariance_matrix.csv",
                mime="text/csv",
            )

        with corr_tab:
            st.dataframe(
                correlation_matrix.round(4),
                use_container_width=True,
            )

            fig = px.imshow(
                correlation_matrix,
                text_auto=".2f",
                aspect="auto",
                zmin=-1,
                zmax=1,
                title="Stock Return Correlations",
            )

            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "The covariance matrix uses daily percentage returns and is "
            "not annualised. The government security is excluded because "
            "its actual daily market prices are not loaded by this app."
        )

    else:
        st.info(
            "At least two securities with usable price histories are "
            "needed to calculate a covariance matrix."
        )

    st.subheader("Portfolio Risk Measures")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric("Daily Standard Deviation", safe_number(portfolio["daily_std"], 6))
    r2.metric("Daily Variance", safe_number(portfolio["daily_variance"], 8))
    r3.metric("Annualised Volatility", percentage(portfolio["annual_volatility"]))
    r4.metric("Sharpe Ratio", safe_number(sharpe_ratio, 3))

    st.subheader("Benchmark-Based Performance Measures")

    if not benchmark_prices.empty and len(portfolio_value) >= 3:

        benchmark_returns = (
            _clean_index(benchmark_prices)
            .pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
        )

        aligned = pd.concat(
            [
                _clean_index(portfolio_returns).rename("Portfolio"),
                benchmark_returns.rename("Benchmark"),
            ],
            axis=1,
            join="inner",
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if len(aligned) >= 30:

            benchmark_variance = aligned["Benchmark"].var(ddof=1)

            if pd.notna(benchmark_variance) and benchmark_variance > 1e-12:

                beta = (
                    aligned["Portfolio"].cov(aligned["Benchmark"])
                    / benchmark_variance
                )

                periods = len(aligned)

                portfolio_growth = float((1 + aligned["Portfolio"]).prod())
                benchmark_growth = float((1 + aligned["Benchmark"]).prod())

                annual_portfolio_return = (
                    portfolio_growth ** (252 / periods) - 1
                    if portfolio_growth > 0
                    else np.nan
                )

                annual_benchmark_return = (
                    benchmark_growth ** (252 / periods) - 1
                    if benchmark_growth > 0
                    else np.nan
                )

                rf = rf_annual / 100

                jensen_alpha = (
                    annual_portfolio_return
                    - (rf + beta * (annual_benchmark_return - rf))
                ) * 100

                treynor = (
                    (annual_portfolio_return - rf) / beta
                    if pd.notna(annual_portfolio_return)
                    and abs(beta) > 1e-8
                    else np.nan
                )

                matched_std = aligned["Portfolio"].std(ddof=1)
                matched_excess = aligned["Portfolio"] - daily_rf

                matched_sharpe = (
                    matched_excess.mean() / matched_std * np.sqrt(252)
                    if pd.notna(matched_std) and matched_std > 1e-12
                    else np.nan
                )

                m1, m2, m3, m4 = st.columns(4)

                m1.metric("Portfolio Beta", safe_number(beta, 3))
                m2.metric("Jensen's Alpha", percentage(jensen_alpha))
                m3.metric("Treynor Ratio", safe_number(treynor, 4))
                m4.metric("Matched-Period Sharpe Ratio", safe_number(matched_sharpe, 3))

                st.caption(
                    f"Metrics use {len(aligned)} overlapping daily "
                    f"observations for {benchmark_name}. Jensen's Alpha "
                    "is shown as an annual percentage. Treynor Ratio is "
                    "annual excess return divided by beta."
                )

            else:
                st.info(
                    "Benchmark variance is zero or unavailable. Beta and "
                    "the associated performance measures cannot be estimated."
                )

        else:
            st.info(
                f"Only {len(aligned)} overlapping daily observations were "
                "found. At least 30 are required for these benchmark-based "
                "metrics. Select a wider date range."
            )

    else:
        st.info(
            "Benchmark data is unavailable for the selected period. "
            "Try another date range or refresh market data."
        )

    st.caption(
        "The portfolio history holds the Government Security at cost. "
        "Its actual daily bond return is not included because verified bond "
        "price and cash-flow data are unavailable. The risk-free rate above "
        "is an assumption for the performance ratios, not a measured return "
        "on the portfolio's Government Security."
    )


# ============================================================
# CAPM ANALYSIS
# ============================================================

elif page == "CAPM Analysis":

    st.title("CAPM Analysis")

    st.write(
        "The Capital Asset Pricing Model relates a security's expected "
        "return to its sensitivity (beta) to the overall market. Beta and "
        "alpha are estimated by regressing daily excess returns of the "
        "security on daily excess returns of the market benchmark."
    )

    st.latex(r"R_i - R_f = \alpha + \beta\,(R_m - R_f) + \epsilon")
    st.latex(r"E(R_i) = R_f + \beta\,\big(E(R_m) - R_f\big)")

    c1, c2, c3 = st.columns(3)

    with c1:
        capm_benchmark = st.selectbox(
            "Market benchmark",
            list(BENCHMARKS.keys()),
            key="capm_benchmark",
        )

    with c2:
        capm_rf = st.number_input(
            "Annual risk-free rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=6.03,
            step=0.25,
            key="capm_rf",
            help="Assumption; not a verified market yield.",
        )

    with c3:
        capm_mrp = st.number_input(
            "Expected market risk premium (%)",
            min_value=0.0,
            max_value=20.0,
            value=6.0,
            step=0.5,
            key="capm_mrp",
            help=(
                "Assumed annual E(Rm) − Rf used for the CAPM required "
                "return. This is an assumption you control."
            ),
        )

    market_returns = get_benchmark_returns(
        capm_benchmark, analysis_start, analysis_end
    )

    if market_returns.empty:
        st.error(
            "Benchmark data could not be retrieved for this period. "
            "Try another date range or refresh market data."
        )
        st.stop()

    st.divider()

    target_options = ["Estimated portfolio"] + list(COMPANIES.keys())

    capm_target = st.selectbox(
        "Security to analyse",
        target_options,
        key="capm_target",
    )

    asset_ret = target_returns(capm_target)

    try:
        capm = fit_capm(asset_ret, market_returns, capm_rf)
    except Exception as error:
        st.error(f"CAPM regression could not be completed: {error}")
        capm = None

    if capm is not None:

        st.subheader(f"Regression Results: {capm_target}")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Beta (β)", safe_number(capm["beta"], 3))
        m2.metric("Annualised Alpha (α)", percentage(capm["alpha_annual_pct"]))
        m3.metric("R-squared", safe_number(capm["r_squared"], 3))
        m4.metric("Observations", f'{capm["observations"]}')

        required_return = capm_rf + capm["beta"] * capm_mrp

        m5, m6 = st.columns(2)

        m5.metric(
            "CAPM Required Return",
            percentage(required_return),
            help="Rf + β × market risk premium (annual).",
        )

        realised = annualised_return(
            capm["combined"]["Asset"]
        )

        m6.metric(
            "Realised Annualised Return",
            percentage(realised * 100) if pd.notna(realised) else "N/A",
        )

        stats_df = pd.DataFrame({
            "Parameter": ["Alpha (daily)", "Beta"],
            "Estimate": [capm["alpha_daily"], capm["beta"]],
            "t-statistic": [capm["t_alpha"], capm["t_beta"]],
            "p-value (approx.)": [capm["p_alpha"], capm["p_beta"]],
        })

        st.dataframe(
            stats_df.round(6),
            use_container_width=True,
            hide_index=True,
        )

        # Scatter with fitted line
        data = capm["combined"]
        x = (data["Market"] - capm["daily_rf"]) * 100
        y = (data["Asset"] - capm["daily_rf"]) * 100

        x_line = np.array([x.min(), x.max()])
        y_line = capm["alpha_daily"] * 100 + capm["beta"] * x_line

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", name="Daily observations",
            marker=dict(size=5, opacity=0.5),
        ))

        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines", name="Fitted line",
        ))

        fig.update_layout(
            title=f"{capm_target} vs {capm_benchmark}: Daily Excess Returns",
            xaxis_title=f"{capm_benchmark} excess return (%)",
            yaxis_title=f"{capm_target} excess return (%)",
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---- Beta interpretation (after the chart) ----
        st.subheader("Beta Interpretation")

        b = capm["beta"]

        if b < 0:
            beta_text = (
                f"A beta of **{b:.2f}** is negative: the security has "
                "tended to move **opposite** to the market."
            )
        elif b < 0.9:
            beta_text = (
                f"A beta of **{b:.2f}** is below 1: the security is "
                f"**defensive**. A 1% market move has been associated with "
                f"roughly a {b:.2f}% move in {capm_target}."
            )
        elif b <= 1.1:
            beta_text = (
                f"A beta of **{b:.2f}** is close to 1: the security has "
                "moved **broadly in line with the market**."
            )
        else:
            beta_text = (
                f"A beta of **{b:.2f}** is above 1: the security is "
                f"**aggressive**. A 1% market move has been associated with "
                f"roughly a {b:.2f}% move in {capm_target}, in both "
                "directions."
            )

        r2 = capm["r_squared"]

        if pd.notna(r2):
            fit_text = (
                f"R-squared is **{r2:.2f}**, so about {r2 * 100:.0f}% of the "
                "security's daily return variation is explained by the "
                f"market ({capm_benchmark}); the rest is stock-specific "
                "risk."
            )
        else:
            fit_text = ""

        p_a = capm["p_alpha"]

        if pd.notna(p_a):
            if p_a < 0.05:
                direction = "outperformed" if capm["alpha_daily"] > 0 else "underperformed"
                alpha_text = (
                    f"Annualised alpha is **{capm['alpha_annual_pct']:.2f}%** "
                    f"and statistically significant at the 5% level: after "
                    f"adjusting for market risk, the security {direction} "
                    "what CAPM predicts over this period."
                )
            else:
                alpha_text = (
                    f"Annualised alpha is **{capm['alpha_annual_pct']:.2f}%**, "
                    "but it is **not statistically significant** at the 5% "
                    "level, so it cannot be told apart from zero."
                )
        else:
            alpha_text = ""

        st.markdown(
            f"{beta_text}\n\n{fit_text}\n\n{alpha_text}\n\n"
            f"Under CAPM, with your assumptions, the required annual return "
            f"is **{required_return:.2f}%** "
            f"(= {capm_rf:.2f}% + {b:.2f} × {capm_mrp:.2f}%)."
        )

        st.caption(
            "p-values use a normal approximation, which is reasonable with "
            "30+ observations. Daily-return betas can differ from the 5-year "
            "monthly betas that data providers publish. A low R-squared means "
            "much of the stock's movement is unrelated to the market."
        )

        if capm_target == "Estimated portfolio":
            st.info(
                "The portfolio history holds the Government Security at "
                "cost (zero volatility), which pulls the portfolio beta "
                "toward 0.9× the equity beta."
            )

    # --------------------------------------------------------
    # BETAS FOR ALL STOCKS + SECURITY MARKET LINE
    # --------------------------------------------------------

    st.divider()
    st.subheader("Betas and Security Market Line: All Holdings")

    rows = []

    for name in list(COMPANIES.keys()) + ["Estimated portfolio"]:
        series = target_returns(name)

        if series.empty:
            continue

        try:
            res = fit_capm(series, market_returns, capm_rf)
        except Exception:
            continue

        realised_ann = annualised_return(res["combined"]["Asset"]) * 100

        rows.append({
            "Security": name,
            "Beta": res["beta"],
            "Alpha (annual %)": res["alpha_annual_pct"],
            "R-squared": res["r_squared"],
            "CAPM Required Return (%)": capm_rf + res["beta"] * capm_mrp,
            "Realised Return (%)": realised_ann,
        })

    if rows:

        beta_df = pd.DataFrame(rows)

        st.dataframe(
            beta_df.round(3),
            use_container_width=True,
            hide_index=True,
        )

        beta_max = max(2.0, float(beta_df["Beta"].max()) + 0.2)
        beta_min = min(0.0, float(beta_df["Beta"].min()) - 0.1)
        beta_grid = np.array([beta_min, beta_max])

        sml = go.Figure()

        sml.add_trace(go.Scatter(
            x=beta_grid,
            y=capm_rf + beta_grid * capm_mrp,
            mode="lines",
            name="Security Market Line",
        ))

        sml.add_trace(go.Scatter(
            x=beta_df["Beta"],
            y=beta_df["Realised Return (%)"],
            mode="markers+text",
            text=beta_df["Security"],
            textposition="top center",
            name="Realised return",
        ))

        sml.update_layout(
            title="Security Market Line vs Realised Returns",
            xaxis_title="Beta",
            yaxis_title="Annual return (%)",
        )

        st.plotly_chart(sml, use_container_width=True)

        st.caption(
            "Points above the line earned more than CAPM required for their "
            "risk over this period (positive alpha); points below earned "
            "less. Realised returns over short windows are noisy and do not "
            "predict future performance."
        )

    else:
        st.info("Not enough data to estimate betas for the selected period.")


# ============================================================
# SUMMARY: 1 / 3 / 5 YEAR MASTER TABLES
# ============================================================

elif page == "Summary":

    st.title("Summary")
    st.caption(
        "Master tables of return, risk, CAPM, alpha and Treynor ratio over "
        "the last 1, 3 and 5 years: first for the whole portfolio, then for "
        "each security. The windows always end today and do not depend on "
        "the sidebar date range."
    )

    sc1, sc2 = st.columns(2)

    with sc1:
        summary_benchmark = st.selectbox(
            "Market benchmark",
            list(BENCHMARKS.keys()),
            key="summary_benchmark",
        )

    with sc2:
        summary_rf = st.number_input(
            "Annual risk-free rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=6.03,
            step=0.25,
            key="summary_rf",
            help="Assumption used for CAPM, alpha, Treynor and Sharpe.",
        )

    horizons = [1, 3, 5]
    labels = [f"{h} Year" if h == 1 else f"{h} Years" for h in horizons]
    start_5y = today - timedelta(days=int(365.25 * 5))

    with st.spinner("Building the 1, 3 and 5 year summary..."):
        prices_5y = load_prices(start_5y, today)
        market_5y = get_benchmark_returns(summary_benchmark, start_5y, today)

    if prices_5y.empty:
        st.error("No historical price data could be loaded for the last 5 years.")
        st.stop()

    if market_5y.empty:
        st.error("Benchmark data could not be retrieved for the last 5 years.")
        st.stop()

    def calc_metrics(asset_returns, market_returns):
        """All summary metrics for one return series (daily decimals)."""

        reg = fit_capm(asset_returns, market_returns, summary_rf)

        data = reg["combined"]
        asset, mkt = data["Asset"], data["Market"]

        rf = summary_rf / 100
        beta = reg["beta"]

        port_ann = annualised_return(asset)
        bench_ann = annualised_return(mkt)

        expected = rf + beta * (bench_ann - rf)

        curve = (1 + asset).cumprod()

        return {
            "port_cum": ((1 + asset).prod() - 1) * 100,
            "port_ann": port_ann * 100,
            "bench_ann": bench_ann * 100,
            "excess_ann": (port_ann - bench_ann) * 100,
            "vol": asset.std(ddof=1) * np.sqrt(252) * 100,
            "max_dd": (curve / curve.cummax() - 1).min() * 100,
            "beta": beta,
            "r2": reg["r_squared"],
            "expected": expected * 100,
            "alpha": (port_ann - expected) * 100,
            "treynor": (
                (port_ann - rf) / beta if abs(beta) > 1e-8 else np.nan
            ),
            "sharpe": (
                (asset - reg["daily_rf"]).mean()
                / asset.std(ddof=1)
                * np.sqrt(252)
            ),
            "obs": reg["observations"],
        }

    # ========================================================
    # PORTFOLIO MASTER TABLE
    # ========================================================

    results = {}
    notes = []

    for h, label in zip(horizons, labels):
        win_start = pd.Timestamp(today - timedelta(days=int(365.25 * h)))

        win_prices = prices_5y[prices_5y.index >= win_start]
        win_market = market_5y[market_5y.index >= win_start]

        try:
            if len(win_prices) < 40:
                raise ValueError("not enough price history in this window.")

            pf_h = build_portfolio(win_prices, weights)

            if pf_h is None:
                raise ValueError("the portfolio could not be built.")

            results[label] = calc_metrics(pf_h["portfolio_returns"], win_market)

            late = []

            for name, w in weights.items():
                if w <= 0:
                    continue

                if name not in win_prices.columns:
                    late.append(f"{name} (no data)")
                    continue

                first_valid = win_prices[name].first_valid_index()

                if first_valid is None:
                    late.append(f"{name} (no data)")
                elif pd.Timestamp(first_valid) > win_start + pd.Timedelta(days=7):
                    late.append(f"{name} (from {pd.Timestamp(first_valid):%b %Y})")

            if late:
                notes.append(
                    f"**{label}:** limited history for " + ", ".join(late)
                    + ". In the portfolio those allocations are held at cost "
                    "until their first available price, which understates "
                    "volatility."
                )

        except Exception as error:
            results[label] = None
            notes.append(f"**{label}:** could not be calculated ({error})")

    def fmt_cell(value, kind):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "N/A"
        if kind == "pct":
            return percentage(value)
        if kind == "n3":
            return safe_number(value, 3)
        if kind == "n4":
            return safe_number(value, 4)
        if kind == "int":
            return f"{int(value)}"
        return str(value)

    row_spec = [
        ("Return", "Portfolio return, cumulative (%)", "port_cum", "pct"),
        ("", "Portfolio return, annualised (%)", "port_ann", "pct"),
        ("", f"{summary_benchmark} return, annualised (%)", "bench_ann", "pct"),
        ("", "Excess return vs benchmark (% p.a.)", "excess_ann", "pct"),
        ("Risk", "Annualised volatility (%)", "vol", "pct"),
        ("", "Maximum drawdown (%)", "max_dd", "pct"),
        ("CAPM", "Beta (β)", "beta", "n3"),
        ("", "R-squared", "r2", "n3"),
        ("", "CAPM expected return (% p.a.)", "expected", "pct"),
        ("", "Jensen's alpha (% p.a.)", "alpha", "pct"),
        ("Risk-adjusted", "Treynor ratio", "treynor", "n4"),
        ("", "Sharpe ratio", "sharpe", "n3"),
        ("Data", "Daily observations", "obs", "int"),
    ]

    master = pd.DataFrame({
        "Category": [r[0] for r in row_spec],
        "Metric": [r[1] for r in row_spec],
        **{
            label: [
                fmt_cell(results[label][r[2]] if results[label] else None, r[3])
                for r in row_spec
            ]
            for label in labels
        },
    })

    st.subheader("Portfolio Master Table")

    st.dataframe(master, use_container_width=True, hide_index=True)

    numeric_master = pd.DataFrame({
        "Metric": [r[1] for r in row_spec],
        **{
            label: [
                results[label][r[2]] if results[label] else np.nan
                for r in row_spec
            ]
            for label in labels
        },
    })

    st.download_button(
        "Download Portfolio Master Table",
        data=numeric_master.to_csv(index=False).encode("utf-8"),
        file_name="portfolio_master_summary.csv",
        mime="text/csv",
    )

    for note in notes:
        st.warning(note)

    chart_labels = [l for l in labels if results[l]]

    if chart_labels:
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=chart_labels,
            y=[results[l]["port_ann"] for l in chart_labels],
            name="SmartVest Portfolio",
        ))

        fig.add_trace(go.Bar(
            x=chart_labels,
            y=[results[l]["bench_ann"] for l in chart_labels],
            name=summary_benchmark,
        ))

        fig.update_layout(
            barmode="group",
            title="Annualised Return: Portfolio vs Benchmark",
            yaxis_title="% per year",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("How to read it")

        lines = []

        for l in chart_labels:
            a = results[l]["alpha"]

            if pd.isna(a):
                continue

            lines.append(
                f"- **{l}:** Jensen's alpha of {a:+.2f}% a year, so the "
                f"portfolio {'earned more' if a > 0 else 'earned less'} than "
                f"CAPM expected for a beta of {results[l]['beta']:.2f}."
            )

        st.markdown("\n".join(lines))

    # ========================================================
    # SECURITY-WISE MASTER TABLE
    # ========================================================

    st.divider()
    st.subheader("Security-wise Master Table")

    st.caption(
        "Companies down the side, every metric across the top. Click a "
        "column header to sort."
    )

    sec_columns = [
        ("port_cum", "Cumulative Return (%)", "%.2f"),
        ("port_ann", "Annualised Return (%)", "%.2f"),
        ("bench_ann", f"{summary_benchmark} Annualised Return (%)", "%.2f"),
        ("excess_ann", "Excess vs Benchmark (% p.a.)", "%.2f"),
        ("vol", "Annualised Volatility (%)", "%.2f"),
        ("max_dd", "Max Drawdown (%)", "%.2f"),
        ("beta", "Beta (β)", "%.3f"),
        ("r2", "R-squared", "%.3f"),
        ("expected", "CAPM Expected Return (% p.a.)", "%.2f"),
        ("alpha", "Jensen's Alpha (% p.a.)", "%.2f"),
        ("treynor", "Treynor Ratio", "%.4f"),
        ("sharpe", "Sharpe Ratio", "%.3f"),
        ("obs", "Observations", "%.0f"),
    ]

    sec_config = {
        label: st.column_config.NumberColumn(label, format=fmt)
        for _, label, fmt in sec_columns
    }

    sec_tables = {}
    sec_notes = {}

    for h, label in zip(horizons, labels):
        win_start = pd.Timestamp(today - timedelta(days=int(365.25 * h)))

        win_prices = prices_5y[prices_5y.index >= win_start]
        win_market = market_5y[market_5y.index >= win_start]

        rows = []
        late = []

        if results.get(label):
            rows.append({
                "Security": "SmartVest Portfolio",
                "Data From": f"{win_prices.index[0]:%d %b %Y}",
                **results[label],
            })

        for name in COMPANIES:
            row = {"Security": name, "Data From": "N/A"}

            if name in win_prices.columns:
                s = win_prices[name].dropna()

                if len(s) >= 2:
                    row["Data From"] = f"{s.index[0]:%d %b %Y}"

                    if s.index[0] > win_start + pd.Timedelta(days=7):
                        late.append(f"{name} (from {s.index[0]:%b %Y})")

                    rets = (
                        s.pct_change(fill_method=None)
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna()
                    )

                    try:
                        row.update(calc_metrics(rets, win_market))
                    except Exception:
                        pass

            rows.append(row)

        table = pd.DataFrame(rows)

        for key, _, _ in sec_columns:
            if key not in table.columns:
                table[key] = np.nan

        table = table[["Security", "Data From"] + [k for k, _, _ in sec_columns]]
        table = table.rename(columns={k: lbl for k, lbl, _ in sec_columns})

        sec_tables[label] = table
        sec_notes[label] = late

    sec_tabs = st.tabs(labels)

    for tab, label in zip(sec_tabs, labels):
        with tab:
            st.dataframe(
                sec_tables[label],
                column_config=sec_config,
                use_container_width=True,
                hide_index=True,
            )

            if sec_notes[label]:
                st.caption(
                    "Shorter history than this window (figures cover only the "
                    "period since the first available price): "
                    + ", ".join(sec_notes[label]) + "."
                )

            na_names = sec_tables[label].loc[
                sec_tables[label]["Beta (β)"].isna(), "Security"
            ].tolist()

            if na_names:
                st.caption(
                    "Not enough overlapping daily data (30 needed) for: "
                    + ", ".join(na_names) + "."
                )

    all_sec = pd.concat(
        [t.assign(Horizon=label) for label, t in sec_tables.items()],
        ignore_index=True,
    )

    all_sec = all_sec[["Horizon"] + [c for c in all_sec.columns if c != "Horizon"]]

    st.download_button(
        "Download Security-wise Master Table (all horizons)",
        data=all_sec.to_csv(index=False).encode("utf-8"),
        file_name="security_master_summary.csv",
        mime="text/csv",
    )

    st.caption(
        "Returns, volatility, beta, alpha and Treynor ratio use daily "
        "returns (adjusted closing prices) on the dates both the security "
        "or portfolio and the benchmark traded. CAPM expected return = Rf + "
        "β × (benchmark annualised return − Rf), using the benchmark's "
        "realised return over the same window. Jensen's alpha = annualised "
        "return − CAPM expected return. Treynor ratio = (annualised return "
        "− Rf) ÷ beta, shown as a decimal, and becomes unreliable when beta "
        "is close to zero or negative. The portfolio assumes it was bought "
        "at the start of each window with today's weights and never "
        "rebalanced; its Government Security is held at cost (zero "
        "volatility), which lowers volatility and beta. Historical results "
        "do not predict future returns."
    )


# ============================================================
# INVESTMENT SIMULATOR
# ============================================================

elif page == "Investment Simulator":

    st.title("🔮 Investment Simulator")

    st.write(
        "Use historical data to see what an investment would have become, "
        "or use the CAPM-based expected portfolio return to model a "
        "future scenario."
    )

    sim_mode = st.radio(
        "Choose Simulation",
        ["Historical — What If?", "CAPM-Based Future Projection"],
        horizontal=True,
    )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _rebase(series, amount):
        s = series.dropna()
        return s / float(s.iloc[0]) * amount

    def _naive(ts):
        ts = pd.Timestamp(ts)
        return ts.tz_localize(None) if ts.tzinfo is not None else ts

    def run_strategy(strategy, start, end, amount):
        """Value history of `amount` invested in `strategy` from `start`."""

        if strategy == "SmartVest Portfolio":
            hist_prices = load_prices(start, end)

            if hist_prices.empty:
                raise ValueError(
                    "No price data could be loaded for that date range."
                )

            pf = build_portfolio(hist_prices, weights)

            if pf is None:
                raise ValueError("The portfolio could not be built.")

            values = _clean_index(pf["portfolio_values"].dropna())
            scale = amount / float(values.iloc[0])

            return {
                "values": values * scale,
                "portfolio": pf,
                "scale": scale,
                "prices": hist_prices,
            }

        if strategy in COMPANIES:
            hist_prices = load_prices(start, end)

            if strategy not in hist_prices.columns:
                raise ValueError(
                    f"No price data is available for {strategy} "
                    "in that date range."
                )

            return {
                "values": _clean_index(
                    _rebase(hist_prices[strategy], amount)
                ),
                "portfolio": None,
                "scale": None,
                "prices": hist_prices,
            }

        series = load_benchmark(BENCHMARKS[strategy], start, end)

        if series.empty:
            raise ValueError(
                f"{strategy} data could not be retrieved for that range."
            )

        return {
            "values": _clean_index(_rebase(series, amount)),
            "portfolio": None,
            "scale": None,
            "prices": None,
        }

    def project(annual_rate, initial, monthly, years):
        """Year-end values with monthly compounding and contributions."""

        monthly_rate = (1 + annual_rate) ** (1 / 12) - 1
        value = initial
        out = [value]

        for month in range(1, years * 12 + 1):
            value = value * (1 + monthly_rate) + monthly
            if month % 12 == 0:
                out.append(value)

        return out

    # ========================================================
    # HISTORICAL — WHAT IF?
    # ========================================================

    if sim_mode == "Historical — What If?":

        st.subheader("📅 What would my investment be worth today?")

        h1, h2, h3 = st.columns(3)

        with h1:
            hist_amount = st.number_input(
                "Initial Investment (₹)",
                min_value=1000,
                max_value=1_000_000_000,
                value=10_000_000,
                step=100_000,
            )

        with h2:
            hist_date = st.date_input(
                "Investment Date",
                value=today - timedelta(days=1826),
                min_value=date(2010, 1, 1),
                max_value=today - timedelta(days=1),
                format="YYYY/MM/DD",
            )

        with h3:
            hist_strategy = st.selectbox(
                "Investment Strategy",
                ["SmartVest Portfolio"]
                + list(COMPANIES.keys())
                + list(BENCHMARKS.keys()),
            )

        st.caption(
            "Historical simulation uses Yahoo Finance adjusted closing "
            "prices. It does not include taxes, brokerage, transaction "
            "costs or portfolio rebalancing between the selected start "
            "date and today. The Government Security is held at cost."
        )

        if st.button("Calculate Historical Value", type="primary"):

            try:
                with st.spinner("Calculating historical value..."):
                    amount = float(hist_amount)

                    run = run_strategy(
                        hist_strategy, hist_date, today, amount
                    )
                    values = run["values"]

                    compare = None

                    if hist_strategy != "Nifty 50":
                        nifty = load_benchmark(
                            BENCHMARKS["Nifty 50"], hist_date, today
                        )
                        if not nifty.empty:
                            compare = _clean_index(_rebase(nifty, amount))

                    notes = []

                    actual_start = values.index[0]

                    if actual_start > _naive(hist_date) + pd.Timedelta(days=7):
                        notes.append(
                            f"Price data for {hist_strategy} starts on "
                            f"{actual_start:%d %b %Y}, so the simulation "
                            "begins on that date instead."
                        )

                    breakdown = None

                    if run["portfolio"] is not None:
                        pf = run["portfolio"]
                        scale = run["scale"]
                        hp = run["prices"]

                        late = []
                        rows = []

                        for name in COMPANIES:
                            first_date = (
                                hp[name].first_valid_index()
                                if name in hp.columns
                                else None
                            )

                            if weights[name] > 0:
                                if first_date is None:
                                    late.append(f"{name} (no data)")
                                elif _naive(first_date) > (
                                    _naive(hist_date) + pd.Timedelta(days=7)
                                ):
                                    late.append(
                                        f"{name} (from {_naive(first_date):%d %b %Y})"
                                    )

                            rows.append({
                                "Security": name,
                                "Weight (%)": weights[name],
                                "Invested (₹)": pf["starting_values"][name] * scale,
                                "Value Today (₹)": pf["ending_values"][name] * scale,
                                "Return (%)": pf["period_returns"].get(
                                    name, np.nan
                                ),
                            })

                        rows.append({
                            "Security": GS_NAME,
                            "Weight (%)": GS_WEIGHT,
                            "Invested (₹)": GS_INVESTMENT * scale,
                            "Value Today (₹)": GS_INVESTMENT * scale,
                            "Return (%)": np.nan,
                        })

                        breakdown = pd.DataFrame(rows)

                        if late:
                            notes.append(
                                "No price history on the investment date for: "
                                + ", ".join(late)
                                + ". Those allocations are held at cost until "
                                "their first available price."
                            )

                        if pf["missing"]:
                            notes.append(
                                "No usable price series for: "
                                + ", ".join(pf["missing"])
                                + ". Those allocations are held at cost."
                            )

                        if not valid_allocation:
                            notes.append(
                                f"Your sidebar equity weights total "
                                f"{total_equity_weight:.1f}%, not 90%. The "
                                "result is scaled to your investment amount."
                            )

                    st.session_state["hist_result"] = {
                        "strategy": hist_strategy,
                        "amount": amount,
                        "start": hist_date,
                        "values": values,
                        "compare": compare,
                        "breakdown": breakdown,
                        "notes": notes,
                    }

            except Exception as error:
                st.session_state.pop("hist_result", None)
                st.error(f"Could not calculate the historical value: {error}")

        res = st.session_state.get("hist_result")

        if res is not None:

            values = res["values"]
            amount = res["amount"]

            final_value = float(values.iloc[-1])
            gain = final_value - amount
            gain_pct = gain / amount * 100

            years_held = (values.index[-1] - values.index[0]).days / 365.25

            cagr = (
                ((final_value / amount) ** (1 / years_held) - 1) * 100
                if years_held >= 0.25 and final_value > 0
                else np.nan
            )

            max_drawdown = (values / values.cummax() - 1).min() * 100

            st.divider()
            st.subheader("Result")

            st.caption(
                f"{money(amount)} invested in {res['strategy']} on "
                f"{res['start']:%d %b %Y}, valued on "
                f"{values.index[-1]:%d %b %Y}."
            )

            for note in res["notes"]:
                st.warning(note)

            k1, k2, k3, k4, k5 = st.columns(5)

            k1.metric("Invested", money(amount))
            k2.metric("Value Today", money(final_value))
            k3.metric("Gain / Loss", money(gain), f"{gain_pct:.2f}%")
            k4.metric("CAGR", percentage(cagr))
            k5.metric("Max Drawdown", percentage(max_drawdown))

            chart = values.rename(res["strategy"]).to_frame()

            if res["compare"] is not None:
                chart = pd.concat(
                    [chart, res["compare"].rename("Nifty 50")],
                    axis=1,
                    join="inner",
                )

            st.line_chart(chart)

            if res["breakdown"] is not None:
                with st.expander("See how the money was split"):
                    st.dataframe(
                        res["breakdown"],
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "The Government Security is held at cost, so no "
                        "return is shown for it."
                    )

            st.caption(
                "Past performance does not guarantee future results. "
                "CAGR is the compound annual growth rate over the holding "
                "period; Max Drawdown is the largest peak-to-trough fall."
            )

    # ========================================================
    # CAPM-BASED FUTURE PROJECTION
    # ========================================================

    else:

        st.subheader("🚀 Where could my investment be in the future?")

        a1, a2, a3 = st.columns(3)

        with a1:
            fut_amount = st.number_input(
                "Initial Investment (₹)",
                min_value=1000,
                max_value=1_000_000_000,
                value=10_000_000,
                step=100_000,
                key="fut_amount",
            )

        with a2:
            fut_years = st.slider(
                "Time Horizon (years)",
                min_value=1,
                max_value=30,
                value=10,
                key="fut_years",
            )

        with a3:
            fut_strategy = st.selectbox(
                "Investment Strategy",
                ["SmartVest Portfolio"] + list(COMPANIES.keys()),
                key="fut_strategy",
            )

        b1, b2, b3 = st.columns(3)

        with b1:
            fut_monthly = st.number_input(
                "Monthly Contribution (₹)",
                min_value=0,
                max_value=10_000_000,
                value=0,
                step=1000,
                key="fut_monthly",
            )

        with b2:
            fut_window = st.selectbox(
                "Beta Estimation Window",
                ["1 year", "3 years", "5 years"],
                index=1,
                key="fut_window",
                help="How much recent history is used to estimate beta.",
            )

        with b3:
            fut_benchmark = st.selectbox(
                "Market Benchmark",
                list(BENCHMARKS.keys()),
                key="fut_benchmark",
            )

        d1, d2, d3 = st.columns(3)

        with d1:
            fut_rf = st.number_input(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=6.03,
                step=0.25,
                key="fut_rf",
            )

        with d2:
            fut_mrp = st.number_input(
                "Market Risk Premium (%)",
                min_value=0.0,
                max_value=20.0,
                value=6.0,
                step=0.5,
                key="fut_mrp",
                help="Assumed annual E(Rm) − Rf.",
            )

        with d3:
            fut_spread = st.number_input(
                "Scenario Spread (± % on premium)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.5,
                key="fut_spread",
                help="Low and High scenarios shift the market premium "
                     "down and up by this amount.",
            )

        st.caption(
            "Expected annual return = Risk-free rate + Beta × Market risk "
            "premium. Beta is estimated from the selected window and assumed "
            "constant. These are hypothetical scenarios, not forecasts."
        )

        if st.button("Calculate Future Projection", type="primary"):

            try:
                with st.spinner("Estimating beta and projecting..."):
                    window_years = int(fut_window.split()[0])
                    win_start = today - timedelta(
                        days=int(365.25 * window_years)
                    )

                    win_prices = load_prices(win_start, today)

                    if win_prices.empty:
                        raise ValueError(
                            "No price data could be loaded for the beta "
                            "estimation window."
                        )

                    if fut_strategy == "SmartVest Portfolio":
                        win_pf = build_portfolio(win_prices, weights)
                        asset_returns = win_pf["portfolio_returns"]
                    else:
                        if fut_strategy not in win_prices.columns:
                            raise ValueError(
                                f"No price data for {fut_strategy} in the "
                                "estimation window."
                            )
                        asset_returns = (
                            win_prices[fut_strategy]
                            .dropna()
                            .pct_change(fill_method=None)
                            .replace([np.inf, -np.inf], np.nan)
                            .dropna()
                        )

                    market = get_benchmark_returns(
                        fut_benchmark, win_start, today
                    )

                    if market.empty:
                        raise ValueError(
                            "Benchmark data could not be retrieved."
                        )

                    reg = fit_capm(asset_returns, market, fut_rf)
                    beta = reg["beta"]

                    rates = {
                        "Low": (fut_rf + beta * max(fut_mrp - fut_spread, 0)) / 100,
                        "Expected": (fut_rf + beta * fut_mrp) / 100,
                        "High": (fut_rf + beta * (fut_mrp + fut_spread)) / 100,
                    }

                    if any(r <= -1 for r in rates.values()):
                        raise ValueError(
                            "A scenario return is at or below -100%, so "
                            "compounding is not meaningful."
                        )

                    initial = float(fut_amount)
                    monthly = float(fut_monthly)

                    projection = pd.DataFrame(
                        {
                            f"{label} ({rate * 100:.2f}% p.a.)": project(
                                rate, initial, monthly, fut_years
                            )
                            for label, rate in rates.items()
                        },
                        index=pd.Index(range(fut_years + 1), name="Year"),
                    )

                    st.session_state["fut_result"] = {
                        "strategy": fut_strategy,
                        "beta": beta,
                        "r_squared": reg["r_squared"],
                        "observations": reg["observations"],
                        "rates": rates,
                        "projection": projection,
                        "initial": initial,
                        "monthly": monthly,
                        "years": fut_years,
                    }

            except Exception as error:
                st.session_state.pop("fut_result", None)
                st.error(f"Could not calculate the projection: {error}")

        res = st.session_state.get("fut_result")

        if res is not None:

            projection = res["projection"]
            total_in = res["initial"] + res["monthly"] * 12 * res["years"]

            expected_col = [c for c in projection.columns if c.startswith("Expected")][0]
            low_col = [c for c in projection.columns if c.startswith("Low")][0]
            high_col = [c for c in projection.columns if c.startswith("High")][0]

            expected_end = float(projection[expected_col].iloc[-1])

            st.divider()
            st.subheader("Projection")

            st.caption(
                f"{res['strategy']} over {res['years']} years, based on "
                f"{res['observations']} daily observations."
            )

            if res["strategy"] == "SmartVest Portfolio":
                st.info(
                    "The portfolio's Government Security is held at cost "
                    "(zero volatility), which lowers the estimated portfolio "
                    "beta."
                )

            k1, k2, k3, k4 = st.columns(4)

            k1.metric("Estimated Beta", safe_number(res["beta"], 3))

            k2.metric(
                "Expected Annual Return",
                percentage(res["rates"]["Expected"] * 100),
            )

            k3.metric("Total Invested", money(total_in))

            k4.metric(
                "Expected Ending Value",
                money(expected_end),
                f"{(expected_end / total_in - 1) * 100:.2f}% on invested",
            )

            st.line_chart(projection)

            scenario_df = pd.DataFrame({
                "Scenario": ["Low", "Expected", "High"],
                "Annual Return (%)": [
                    res["rates"]["Low"] * 100,
                    res["rates"]["Expected"] * 100,
                    res["rates"]["High"] * 100,
                ],
                "Ending Value (₹)": [
                    float(projection[low_col].iloc[-1]),
                    expected_end,
                    float(projection[high_col].iloc[-1]),
                ],
            })

            st.dataframe(
                scenario_df.round(2),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Monthly contributions are added at each month-end. The "
                "projection ignores market volatility, the security's own "
                "alpha, taxes, fees and rebalancing. Low and High scenarios "
                "only shift the market risk premium."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SmartVest is an educational portfolio analysis tool. "
    "Market data may be delayed, unavailable or inaccurate. "
    "Historical returns do not guarantee future results. "
    "This dashboard is not investment advice."
)