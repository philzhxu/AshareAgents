"""East Money (东方财富) market data fetcher for A-stock tickers.

Replaces Yahoo Finance for A-stock OHLCV price data, fundamentals, and
financial statements.  East Money is China's largest financial data portal
and its public JSON APIs (accessed via ``push2his.eastmoney.com`` and
``datacenter.eastmoney.com``) provide complete coverage of Shanghai,
Shenzhen, and Beijing exchange listings — all without authentication.

Uses ``curl_cffi`` for TLS fingerprint impersonation (Chinese financial
sites block standard Python HTTP clients).  Degrades gracefully — returns
the same error-sentinel format as the yfinance functions so the router
layer handles fallback transparently.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime

from .symbol_utils import (
    NoMarketDataError,
    ashare_bare_code,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# East Money API endpoints
# ---------------------------------------------------------------------------
_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_FINANCE_URL = "https://datacenter.eastmoney.com/securities/api/data/v1/get"

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# EMP API standard common params (embedded in every request).
_EMP_UT = "7eea3edcaed734bea9cbfc24409ed989"

# ---------------------------------------------------------------------------
# Symbol / secid helpers
# ---------------------------------------------------------------------------

# Shanghai-listed stocks have codes beginning with "6" (main board + STAR);
# everything else — Shenzhen main board (0xx), SME (2xx), ChiNext (3xx),
# and Beijing (8xx) — uses market code 0 in East Money's secid scheme.
_SHANGHAI_PREFIXES = frozenset({"6"})


def _market_code(bare_code: str) -> int:
    """Return East Money market code: 1 for Shanghai, 0 for others."""
    return 1 if bare_code[:1] in _SHANGHAI_PREFIXES else 0


def _secid(ticker: str) -> str:
    """Convert an A-stock ticker to an East Money ``secid``.

    ``000858.SZ`` → ``0.000858``, ``601318.SS`` → ``1.601318``.
    """
    code = ashare_bare_code(ticker) or ticker.strip().upper()
    return f"{_market_code(code)}.{code}"


def _fetch_json(url: str, params: dict, timeout: float = 15.0) -> dict | None:
    """Fetch a JSON response from an East Money API via ``curl_cffi``.

    Returns the parsed ``data`` key when the response indicates success,
    or ``None`` on any failure.
    """
    try:
        from curl_cffi import requests as cffi_requests
    except ImportError:
        logger.warning("curl_cffi not available for East Money API call")
        return None

    try:
        resp = cffi_requests.get(
            url,
            params=params,
            impersonate="chrome120",
            timeout=timeout,
            headers={
                "User-Agent": _UA,
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept": "application/json, text/html, */*",
                "Referer": "https://quote.eastmoney.com/",
            },
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logger.warning("East Money API call failed (%s): %s", url, exc)
        return None

    # Check for API-level errors.
    if payload.get("rc") not in (None, 0, ""):
        logger.warning("East Money API error (rc=%s): %s", payload.get("rc"), payload.get("message"))
    return payload.get("data") if isinstance(payload, dict) else None


# ---------------------------------------------------------------------------
# OHLCV / K-line data — replaces get_YFin_data_online for A-stocks
# ---------------------------------------------------------------------------

# East Money K-line field mapping:
#   f51 = date, f52 = open, f53 = close, f54 = high, f55 = low,
#   f56 = volume (手, lots of 100 shares),
#   f57 = amount (元), f58 = amplitude%, f59 = change%, f60 = change_amount,
#   f61 = turnover%, f116 = ? (unused)
#
# The EM field order is: date,open,close,high,low,volume,amount,...
# Yahoo Finance order:    Date,Open,High,Low,Close,Adj Close,Volume
# We reorder and add a synthetic Adj Close (= Close).

_KLINE_FIELDS1 = "f1,f2,f3,f4,f5,f6"
_KLINE_FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116"


def _parse_kline_row(raw: str) -> dict:
    """Parse one comma-separated East Money K-line row into a dict.

    East Money field order (``fields2`` parameter):
      f51 = 日期 (date),  f52 = 开盘 (open),   f53 = 收盘 (close),
      f54 = 最高 (high),  f55 = 最低 (low),    f56 = 成交量 (volume, 手),
      f57 = 成交额 (amount), …

    Yahoo Finance / CSV output column order:
      Date, Open, High, Low, Close, Adj Close, Volume

    We reorder the fields and convert volume from 手 (lots) to shares
    so the output is consistent across vendors.
    """
    parts = raw.split(",")
    if len(parts) < 7:
        return {}
    # East Money volume is in 手 (lots of 100 shares); convert to shares.
    volume_lots = float(parts[5]) if len(parts) > 5 else 0.0
    return {
        "Date": parts[0],          # f51
        "Open": parts[1],          # f52
        "High": parts[3],          # f54  (EM order: …open,close,high,low,…)
        "Low": parts[4],           # f55
        "Close": parts[2],         # f53
        "Adj Close": parts[2],     # f53 — EM has no separate adjusted-close
        "Volume": str(int(volume_lots * 100)),  # shares
    }


def get_eastmoney_stock(
    symbol: str,
    start_date: str,
    end_date: str,
    timeout: float = 15.0,
) -> str:
    """Fetch daily OHLCV data from East Money for an A-stock ticker.

    Args:
        symbol: A-stock ticker (e.g. ``000966.SZ``, ``601318.SS``).
        start_date: Start date in ``YYYYMMDD`` or ``YYYY-MM-DD`` format.
        end_date: End date in ``YYYYMMDD`` or ``YYYY-MM-DD`` format.
        timeout: HTTP request timeout in seconds.

    Returns:
        CSV-formatted string with header (same shape as
        ``get_YFin_data_online``), or raises ``NoMarketDataError``.
    """
    # Normalise date format to YYYYMMDD (East Money expects this).
    start = start_date.replace("-", "")
    end = end_date.replace("-", "")

    sec = _secid(symbol)
    params = {
        "fields1": _KLINE_FIELDS1,
        "fields2": _KLINE_FIELDS2,
        "ut": _EMP_UT,
        "klt": "101",       # daily
        "fqt": "1",         # 前复权 (forward-adjusted)
        "secid": sec,
        "beg": start,
        "end": end,
    }

    data = _fetch_json(_KLINE_URL, params, timeout=timeout)
    if data is None or not data.get("klines"):
        raise NoMarketDataError(
            symbol, sec, f"East Money returned no K-line rows for {start_date}–{end_date}"
        )

    rows = [_parse_kline_row(r) for r in data["klines"]]
    rows = [r for r in rows if r]  # drop parse failures
    if not rows:
        raise NoMarketDataError(
            symbol, sec, f"East Money returned unparseable K-line rows for {start_date}–{end_date}"
        )

    # Build CSV output matching yfinance's format.
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"],
    )
    writer.writeheader()
    writer.writerows(rows)

    csv_string = output.getvalue()

    code = ashare_bare_code(symbol) or symbol
    header = (
        f"# Stock data for {code} (East Money / 东方财富) "
        f"from {start_date} to {end_date}\n"
    )
    header += f"# Total records: {len(rows)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    header += "# Note: Volume is converted from 手 to shares (×100). "
    header += "Adj Close = Close (East Money does not provide adjusted close).\n\n"

    return header + csv_string


# ---------------------------------------------------------------------------
# Fundamentals data — replaces get_yfinance_fundamentals for A-stocks
# ---------------------------------------------------------------------------

# East Money financial data fields we extract.  These are sourced from the
# ``RPT_F10_FINANCE_MAINFINADATA`` report which contains the latest quarterly
# and annual financial figures for each listed company.
#
# Field reference (East Money → our label):
_FUNDAMENTAL_FIELD_MAP: dict[str, str] = {
    # Per-share
    "EPSJB":              "EPS (Basic)",
    "EPSXS":              "EPS (Diluted)",
    "BPS":                "Book Value Per Share",
    "MGZBGJ":             "Net Assets Per Share (adjusted)",
    # Revenue & Profit
    "TOTALOPERATEREVE":   "Total Revenue",
    "PARENTNETPROFIT":    "Net Income (Parent)",
    "MLR":                "Gross Profit",
    # Margins
    "XSMLL":              "Net Margin (%)",
    # Returns
    "ROEJQ":              "ROE (%)",
    "ROEKCJQ":            "ROE (Deducted, %)",
    # Balance sheet ratios
    "ZCFZL":              "Debt-to-Asset Ratio (%)",
    "LD":                 "Current Ratio",
    "SD":                 "Quick Ratio",
    # Growth rates (YoY)
    "TOTALOPERATEREVETZ": "Revenue YoY Growth (%)",
    "PARENTNETPROFITTZ":  "Net Income YoY Growth (%)",
    # Per-share cash flows
    "MGJYXJJE":           "Operating CF Per Share",
    "MGWFPLR":            "Undistributed Profit Per Share",
    # Report metadata
    "REPORT_DATE":        "Report Date",
    "REPORT_TYPE":        "Report Type",
    "CURRENCY":           "Currency",
}

# Additional company info fields from a separate lightweight API.
_COMPANY_INFO_URL = "https://push2.eastmoney.com/api/qt/stock/get"

_COMPANY_INFO_FIELDS = (
    "f57,f58,f84,f85,f86,f92,f104,f105,f116,f117,"
    "f162,f163,f164,f167,f168,f169,f170,f171,f173,f174,f175,"
    "f183,f184,f185,f186,f187,f188,f189,f190,f191,f192,f193"
)


def _fetch_company_info(secid: str, timeout: float) -> dict:
    """Fetch lightweight company info (market cap, PE, sector, etc.).

    This API is occasionally unreliable (connection resets); failures are
    logged but do not prevent the rest of the fundamentals report from
    being built.
    """
    try:
        from curl_cffi import requests as cffi_requests
    except ImportError:
        return {}

    params = {
        "fltt": "2",
        "invt": "2",
        "fields": _COMPANY_INFO_FIELDS,
        "secid": secid,
        "ut": _EMP_UT,
    }
    try:
        resp = cffi_requests.get(
            _COMPANY_INFO_URL,
            params=params,
            impersonate="chrome120",
            timeout=timeout,
            headers={
                "User-Agent": _UA,
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept": "application/json, text/html, */*",
                "Referer": "https://quote.eastmoney.com/",
            },
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logger.warning("East Money company info fetch failed: %s", exc)
        return {}

    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    return {}


def _fetch_datacenter(params: dict, timeout: float = 15.0) -> dict | None:
    """Fetch data from the East Money datacenter API.

    The datacenter API wraps results in ``result.data`` (unlike the K-line
    API which uses a top-level ``data`` key).
    """
    try:
        from curl_cffi import requests as cffi_requests
    except ImportError:
        logger.warning("curl_cffi not available for East Money datacenter call")
        return None

    try:
        resp = cffi_requests.get(
            _FINANCE_URL,
            params=params,
            impersonate="chrome120",
            timeout=timeout,
            headers={
                "User-Agent": _UA,
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept": "application/json, text/html, */*",
                "Referer": "https://emweb.securities.eastmoney.com/",
            },
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logger.warning("East Money datacenter call failed: %s", exc)
        return None

    if not isinstance(payload, dict):
        return None
    if not payload.get("success"):
        logger.warning("East Money datacenter returned success=false: %s", payload.get("message"))
        return None
    result = payload.get("result")
    if isinstance(result, dict):
        return result
    return None


def get_eastmoney_fundamentals(
    ticker: str,
    curr_date: str | None = None,
    timeout: float = 15.0,
) -> str:
    """Fetch company fundamentals for an A-stock ticker from East Money.

    Args:
        ticker: A-stock ticker (e.g. ``000966.SZ``).
        curr_date: Current date (used for yfinance compat; not used here).
        timeout: HTTP request timeout in seconds.

    Returns:
        Formatted plaintext fundamentals block, or raises ``NoMarketDataError``.
    """
    sec = _secid(ticker)
    code = ashare_bare_code(ticker) or ticker

    # 1. Fetch the latest quarterly financial data.
    fin_params = {
        "reportName": "RPT_F10_FINANCE_MAINFINADATA",
        "columns": "ALL",
        "filter": f"(SECURITY_CODE=\"{code}\")",
        "pageNumber": "1",
        "pageSize": "3",
        "sortTypes": "-1",
        "sortColumns": "REPORT_DATE",
        "source": "HSF10",
        "client": "PC",
    }

    fin_result = _fetch_datacenter(fin_params, timeout=timeout)
    if fin_result is None:
        raise NoMarketDataError(ticker, sec, "no fundamentals returned from East Money")

    records = fin_result.get("data")
    if not records:
        raise NoMarketDataError(ticker, sec, "no fundamental records returned")

    # 2. Build output.
    # The datacenter record already includes company name and code.
    company_name = records[0].get("SECURITY_NAME_ABBR", code)
    lines = [
        f"# Company Fundamentals: {company_name} ({code}) — East Money / 东方财富",
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Latest financial data section.
    latest = records[0]
    report_date = (latest.get("REPORT_DATE") or "?").split(" ")[0]  # strip time
    report_type = latest.get("REPORT_TYPE", "?")
    lines.append(f"[Latest Financials — {report_type} ({report_date})]")

    for field_id, label in _FUNDAMENTAL_FIELD_MAP.items():
        value = latest.get(field_id)
        if value is not None and field_id not in ("REPORT_DATE", "REPORT_TYPE", "CURRENCY"):
            # Format numbers nicely.
            if isinstance(value, float):
                lines.append(f"{label}: {value:,.4f}")
            else:
                lines.append(f"{label}: {value}")

    # If we have more than one record, show a growth comparison.
    if len(records) > 1:
        prev = records[1]
        prev_date = (prev.get("REPORT_DATE") or "?").split(" ")[0]  # strip time
        lines.append("")
        lines.append(f"[Previous Period — {prev_date}]")
        for field_id, label in _FUNDAMENTAL_FIELD_MAP.items():
            value = prev.get(field_id)
            if value is not None and field_id not in ("REPORT_DATE", "REPORT_TYPE", "CURRENCY"):
                if isinstance(value, float):
                    lines.append(f"{label}: {value:,.4f}")
                else:
                    lines.append(f"{label}: {value}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Financial statements (balance sheet, cash flow, income statement)
# ---------------------------------------------------------------------------
# These are less critical for the immediate A-stock fix — the East Money
# datacenter provides them via other report names — but the function
# signatures exist so the vendor router can be configured to use them.

def get_eastmoney_balance_sheet(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
    timeout: float = 15.0,
) -> str:
    """Fetch balance sheet data from East Money.

    .. note::
        Not yet implemented — East Money datacenter balance-sheet report
        parsing requires additional field mapping.  Falls back gracefully.
    """
    code = ashare_bare_code(ticker) or ticker
    raise NoMarketDataError(
        ticker, code,
        "East Money balance sheet not yet implemented; try another vendor",
    )


def get_eastmoney_cashflow(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
    timeout: float = 15.0,
) -> str:
    """Fetch cash flow data from East Money.

    .. note::
        Not yet implemented — falls back gracefully.
    """
    code = ashare_bare_code(ticker) or ticker
    raise NoMarketDataError(
        ticker, code,
        "East Money cash flow not yet implemented; try another vendor",
    )


def get_eastmoney_income_statement(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
    timeout: float = 15.0,
) -> str:
    """Fetch income statement data from East Money.

    .. note::
        Not yet implemented — falls back gracefully.
    """
    code = ashare_bare_code(ticker) or ticker
    raise NoMarketDataError(
        ticker, code,
        "East Money income statement not yet implemented; try another vendor",
    )
