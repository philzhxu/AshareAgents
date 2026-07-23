"""Symbol normalization and market-data error types for vendor calls.

Yahoo Finance (the default vendor) uses specific ticker conventions that
differ from the broker / TradingView / MT5 style symbols users often type:

    user types        Yahoo wants       why
    ---------------   ---------------   -----------------------------------
    XAUUSD, XAUUSD+   GC=F              gold has no forex pair on Yahoo;
                                        it is quoted as a COMEX future
    EURUSD            EURUSD=X          spot forex pairs take a ``=X`` suffix
    BTCUSD            BTC-USD           crypto pairs use a ``-`` separator
    SPX500, US500     ^GSPC             index CFDs map to Yahoo index symbols

Passing the raw broker symbol to Yahoo returns an empty result, which the
agents previously received as free text and could hallucinate a price
around (see issue #781). Centralizing the mapping here means every yfinance
entry point resolves symbols the same way, and new instruments are added by
appending a table row rather than editing call sites.
"""

from __future__ import annotations

import logging
import re

# NoMarketDataError lives in the vendor-error taxonomy (errors.py); re-exported
# here for the many call sites that import it alongside normalize_symbol.
from .errors import NoMarketDataError as NoMarketDataError

logger = logging.getLogger(__name__)


# ISO-4217 codes common enough to appear in retail forex pairs. A bare
# six-letter symbol whose halves are BOTH in this set is treated as a spot
# forex pair and given Yahoo's ``=X`` suffix.
_FOREX_CURRENCIES = frozenset(
    {
        "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
        "CNY", "CNH", "HKD", "SGD", "SEK", "NOK", "DKK", "PLN",
        "MXN", "ZAR", "TRY", "INR", "KRW", "BRL", "RUB", "THB",
    }
)

# Crypto bases that brokers quote against USD without a separator.
_CRYPTO_BASES = frozenset(
    {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC", "BCH", "DOT", "AVAX", "LINK"}
)

# Explicit aliases for instruments whose broker symbol does not map to a
# Yahoo symbol by rule. Metals/energy resolve to their front-month future;
# index CFD names resolve to the underlying Yahoo index symbol. Extend by
# adding rows — no call site changes required.
_ALIASES = {
    # Precious metals (spot names -> COMEX/NYMEX futures)
    "XAUUSD": "GC=F", "XAU": "GC=F", "GOLD": "GC=F",
    "XAGUSD": "SI=F", "XAG": "SI=F", "SILVER": "SI=F",
    "XPTUSD": "PL=F", "XPDUSD": "PA=F",
    # Energy
    "WTICOUSD": "CL=F", "USOIL": "CL=F", "WTI": "CL=F",
    "BCOUSD": "BZ=F", "UKOIL": "BZ=F", "BRENT": "BZ=F",
    "NATGAS": "NG=F", "XNGUSD": "NG=F",
    "COPPER": "HG=F", "XCUUSD": "HG=F",
    # Index CFDs -> Yahoo index symbols
    "SPX500": "^GSPC", "US500": "^GSPC", "SPX": "^GSPC",
    "NAS100": "^NDX", "US100": "^NDX", "USTEC": "^NDX",
    "US30": "^DJI", "DJI30": "^DJI", "WS30": "^DJI",
    "GER40": "^GDAXI", "GER30": "^GDAXI", "DE40": "^GDAXI",
    "UK100": "^FTSE", "JP225": "^N225", "JPN225": "^N225",
    "FRA40": "^FCHI", "EU50": "^STOXX50E", "HK50": "^HSI",
}

# Yahoo symbols may contain letters, digits, and these structural characters.
_YAHOO_SAFE = re.compile(r"^[A-Za-z0-9._\-\^=]+$")


# Crypto quote currencies that all map to Yahoo's USD pair. Yahoo lists only
# ``<BASE>-USD`` (not the USDT/USDC stablecoin pairs), so a broker symbol quoted
# in any of these resolves to ``-USD`` (#982). Longest first so ``USDT``/``USDC``
# match before the ``USD`` substring.
_CRYPTO_QUOTES = ("USDT", "USDC", "USD")


def crypto_base(raw: str) -> str | None:
    """Return the crypto base (e.g. ``BTC``) for a known USD/USDT/USDC-quoted
    crypto symbol in any form the pipeline may hold — ``BTC-USD``, ``BTCUSD``,
    ``BTC-USDT`` — or None for non-crypto symbols. Purely syntactic.
    """
    if not isinstance(raw, str):
        return None
    compact = raw.strip().upper().rstrip("+").replace("-", "")
    for quote in _CRYPTO_QUOTES:
        if compact.endswith(quote):
            base = compact[: -len(quote)]
            return base if base in _CRYPTO_BASES else None
    return None


def _normalize_crypto(s: str) -> str | None:
    """Return ``<BASE>-USD`` for a known USD/USDT/USDC-quoted crypto, else None."""
    base = crypto_base(s)
    return f"{base}-USD" if base else None


def normalize_symbol(raw: str) -> str:
    """Map a user/broker symbol to its canonical Yahoo Finance symbol.

    Resolution order (first match wins):
      1. Explicit alias table (metals, energy, index CFDs).
      2. Crypto rule: a known crypto base quoted in USD/USDT/USDC (dashed or
         not) -> ``BASE-USD``.
      3. Forex rule: six letters that are two ISO currency codes -> ``PAIR=X``.
      4. Otherwise the upper-cased symbol is returned unchanged (plain
         equities, ETFs, Yahoo-native symbols like ``GC=F`` or ``^GSPC``).

    A trailing ``+`` (broker CFD marker, e.g. ``XAUUSD+``) is stripped before
    matching. The function is purely syntactic — it performs no network
    calls — so it is safe to apply on every request.
    """
    if not isinstance(raw, str) or not raw.strip():
        return raw

    s = raw.strip().upper()
    # Broker CFD/qualifier suffixes Yahoo never uses.
    s = s.rstrip("+")

    crypto = _normalize_crypto(s)
    if s in _ALIASES:
        canonical = _ALIASES[s]
    elif crypto is not None:
        canonical = crypto
    elif len(s) == 6 and s[:3] in _FOREX_CURRENCIES and s[3:] in _FOREX_CURRENCIES:
        canonical = f"{s}=X"
    else:
        canonical = s

    if canonical != raw.strip().upper():
        logger.info("Resolved symbol %r to Yahoo symbol %r", raw, canonical)
    return canonical


def is_yahoo_safe(symbol: str) -> bool:
    """True when ``symbol`` only contains characters Yahoo symbols use."""
    return bool(symbol) and _YAHOO_SAFE.fullmatch(symbol) is not None


# ---------------------------------------------------------------------------
# A-share (China A-stock) ticker utilities
# ---------------------------------------------------------------------------

# Map the Yahoo-style exchange suffix to the lowercase prefix used by
# Sina Finance and other Chinese financial portals in their URLs.
# Note: both ".SS" (Yahoo) and ".SH" (Wind/Tushare/同花顺) map to "sh".
_ASHARE_EXCHANGE_PREFIX: dict[str, str] = {
    ".SZ": "sz",
    ".SS": "sh",
    ".SH": "sh",
    ".BJ": "bj",
}

# Reverse map for display: lowercase prefix → uppercase suffix.
_ASHARE_PREFIX_TO_SUFFIX: dict[str, str] = {
    v: k for k, v in _ASHARE_EXCHANGE_PREFIX.items()
}


def is_ashare(ticker: str) -> bool:
    """Return ``True`` when *ticker* appears to be an A-stock.

    Detects the exchange suffixes ``.SZ`` (Shenzhen), ``.SS`` / ``.SH``
    (Shanghai), and ``.BJ`` (Beijing / New Third Board).

    >>> is_ashare("000858.SZ")
    True
    >>> is_ashare("601318.SS")
    True
    >>> is_ashare("600619.SH")
    True
    >>> is_ashare("AAPL")
    False
    """
    if not isinstance(ticker, str):
        return False
    return ticker.strip().upper().endswith((".SZ", ".SS", ".SH", ".BJ"))


def ashare_bare_code(ticker: str) -> str:
    """Strip the exchange suffix from an A-stock ticker.

    ``000858.SZ`` → ``000858``, ``601318.SS`` → ``601318``.
    Returns the ticker unchanged when the suffix is not recognised.
    """
    if not isinstance(ticker, str):
        return ticker
    upper = ticker.strip().upper()
    for suffix in _ASHARE_EXCHANGE_PREFIX:
        if upper.endswith(suffix):
            return upper[: -len(suffix)]
    return upper


def ashare_exchange_prefix(ticker: str) -> str:
    """Return the lowercase exchange prefix for a Chinese financial portal URL.

    ``000858.SZ`` → ``"sz"``, ``601318.SS`` → ``"sh"``.
    Returns ``""`` when the suffix is not recognised.
    """
    if not isinstance(ticker, str):
        return ""
    upper = ticker.strip().upper()
    for suffix, prefix in _ASHARE_EXCHANGE_PREFIX.items():
        if upper.endswith(suffix):
            return prefix
    return ""


def resolve_cn_stock_name(ticker: str, timeout: float = 10.0) -> str | None:
    """Resolve the Chinese stock name for an A-stock ticker via East Money.

    East Money (东方财富) provides fast, free, no-auth access to Chinese stock
    names.  This is the preferred source for report naming because yfinance
    only returns romanised / English names for A-stocks.

    ``000858.SZ`` → ``"五粮液"``, ``001367.SZ`` → ``"海森药业"``.

    Args:
        ticker: A-stock ticker with exchange suffix (e.g. ``001367.SZ``).
        timeout: HTTP request timeout in seconds.

    Returns:
        The Chinese stock name as a string, or ``None`` on any failure
        (network error, unrecognised ticker, API format change).
    """
    if not is_ashare(ticker):
        return None

    bare = ashare_bare_code(ticker)
    prefix = ashare_exchange_prefix(ticker)

    # East Money market ID: 0=Shenzhen/Beijing, 1=Shanghai
    _EM_MARKET: dict[str, str] = {"sz": "0", "sh": "1", "bj": "0"}
    market = _EM_MARKET.get(prefix, "0")
    secid = f"{market}.{bare}"

    url = (
        "https://push2.eastmoney.com/api/qt/stock/get"
        f"?secid={secid}&fields=f57,f58"
    )

    try:
        from urllib.request import Request, urlopen
        import json as _json

        req = Request(url, headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://quote.eastmoney.com/",
        })
        with urlopen(req, timeout=timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8", errors="replace"))

        name = (data.get("data") or {}).get("f58")
        if name and isinstance(name, str):
            # East Money sometimes inserts spaces between Chinese characters
            # (e.g. "五 粮 液").  Remove interstitial whitespace.
            import re as _re

            cleaned = _re.sub(r"(?<=[一-鿿])\s+(?=[一-鿿])", "", name.strip())
            return cleaned if cleaned else None
        return None
    except Exception:
        logger.debug("Could not resolve Chinese stock name for %s", ticker)
        return None
