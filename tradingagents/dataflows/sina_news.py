"""Sina Finance (新浪财经) stock news fetcher for A-stock tickers.

Sina Finance is one of China's largest financial portals and provides
per-stock news pages that are accessible without authentication. This
module fetches and parses those pages to produce a formatted news block
suitable for prompt injection in the sentiment analyst.

The module uses ``curl_cffi`` to impersonate a Chrome browser TLS
fingerprint — Chinese financial sites commonly block standard Python
HTTP clients (``urllib``, ``requests``) at the TLS layer.

No API key required.  Returns formatted plaintext blocks ready for prompt
injection and degrades gracefully — returns a placeholder string rather than
raising, so callers never special-case missing data.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

from .symbol_utils import ashare_bare_code, ashare_exchange_prefix

logger = logging.getLogger(__name__)

# Sina Finance stock news page template.
# Exchange prefix: "sz" for Shenzhen, "sh" for Shanghai, "bj" for Beijing.
_NEWS_URL = (
    "https://vip.stock.finance.sina.com.cn/corp/go.php/"
    "vCB_AllNewsStock/symbol/{prefix}{code}/displaytype/3.phtml"
)

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _sina_symbol(ticker: str) -> tuple[str, str]:
    """Return ``(exchange_prefix, bare_code)`` for a Sina Finance URL.

    ``000858.SZ`` → ``("sz", "000858")``,
    ``601318.SS`` → ``("sh", "601318")``.
    """
    code = ashare_bare_code(ticker) or ticker.strip().upper()
    prefix = ashare_exchange_prefix(ticker)
    return prefix, code


def _parse_news_html(html: str, ticker: str, limit: int) -> list[dict]:
    """Extract news items from the Sina Finance stock news HTML page.

    The page lists news items inside ``<div class="datelist"><ul>`` as raw
    text lines::

        &nbsp;2026-07-21&nbsp;13:10&nbsp;<a target='_blank' href='URL'>TITLE</a>

    Dates are inline text, not wrapped in a ``<span>``.
    """
    items = []

    # Extract the datelist <ul> block(s).
    ul_blocks = re.findall(
        r'<div class="datelist"[^>]*>.*?<ul>(.*?)</ul>',
        html, re.DOTALL,
    )

    # Each entry: optional whitespace/&nbsp;, date, &nbsp;, time,
    # &nbsp;, <a> tag with href and title.
    entry_pattern = re.compile(
        r'(?:&nbsp;)*\s*(\d{4}-\d{2}-\d{2})\s*&nbsp;\s*(\d{2}:\d{2})\s*'
        r'(?:&nbsp;)*\s*'
        r"<a\s+target='_blank'\s+href='(https?://[^']+)'\s*>"
        r"([^<]+)</a>",
    )

    for block in ul_blocks:
        for match in entry_pattern.finditer(block):
            if len(items) >= limit:
                break
            date_part, time_part, url, title = match.groups()
            title = title.strip()

            # Skip clearly non-stock articles (TV shows, lifestyle, etc.)
            if _is_noise_title(title):
                continue

            items.append({
                "title": title,
                "link": url,
                "date": f"{date_part} {time_part}",
                "source": "新浪财经",
            })

    # Fallback: broader link pattern if datelist not found.
    if not items:
        fallback = re.findall(
            r"<a\s+[^>]*href=['\"](https?://finance\.sina\.com\.cn/[^'\"]+)['\"][^>]*>([^<]+)</a>",
            html,
        )
        for url, title in fallback:
            if len(items) >= limit:
                break
            title = title.strip()
            if len(title) > 10 and not _is_noise_title(title):
                items.append({
                    "title": title,
                    "link": url,
                    "date": "?",
                    "source": "新浪财经",
                })

    return items


def _is_noise_title(title: str) -> bool:
    """Return True if *title* looks like a non-stock lifestyle/spam article."""
    # JavaScript template leakage from the page itself.
    if "+" in title and ("hotstock" in title.lower() or "a[i]" in title):
        return True
    noise_keywords = [
        "奔跑吧", "公益专场", "综艺", "娱乐",
        "爱侣", "情侣", "婚姻", "相亲",
    ]
    return any(kw in title for kw in noise_keywords)


def get_news_sina(
    ticker: str,
    start_date: str,
    end_date: str,
    limit: int = 20,
    timeout: float = 15.0,
) -> str:
    """Fetch recent stock-specific news from Sina Finance.

    Args:
        ticker: A-stock ticker (e.g. ``000858.SZ``, ``601318.SS``).
        start_date: Start date in ``yyyy-mm-dd`` format.
        end_date: End date in ``yyyy-mm-dd`` format.
        limit: Maximum number of articles to return.
        timeout: HTTP request timeout in seconds.

    Returns:
        Formatted plaintext block suitable for prompt injection, or a
        placeholder string when the source is unreachable.
    """
    prefix, code = _sina_symbol(ticker)
    url = _NEWS_URL.format(prefix=prefix, code=code)

    try:
        from curl_cffi import requests as cffi_requests
    except ImportError:
        logger.warning("curl_cffi not available — falling back to urllib for Sina")
        return _get_news_sina_urllib(ticker, url, start_date, end_date, limit, timeout)

    try:
        resp = cffi_requests.get(
            url,
            impersonate="chrome120",
            timeout=timeout,
            headers={
                "User-Agent": _UA,
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:
        logger.warning("Sina Finance news fetch failed for %s: %s", ticker, exc)
        return f"<新浪财经新闻不可用: {type(exc).__name__}>"

    articles = _parse_news_html(html, ticker, limit)

    if not articles:
        return (
            f"<no Sina Finance news found for {ticker} "
            f"between {start_date} and {end_date}>"
        )

    news_str = f"## {ticker} 新闻 (新浪财经), {start_date} 至 {end_date}:\n\n"
    for a in articles:
        news_str += f"### {a['title']} (source: {a['source']})\n"
        news_str += f"Date: {a['date']}\n"
        news_str += f"Link: {a['link']}\n\n"

    return news_str


def _get_news_sina_urllib(
    ticker: str,
    url: str,
    start_date: str,
    end_date: str,
    limit: int,
    timeout: float,
) -> str:
    """Fallback using standard-library ``urllib`` when ``curl_cffi`` is absent."""
    import http.client
    from urllib.request import Request, urlopen

    req = Request(url, headers={"User-Agent": _UA, "Accept-Language": "zh-CN,zh;q=0.9"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Sina Finance (urllib) fetch failed for %s: %s", ticker, exc)
        return f"<新浪财经新闻不可用: {type(exc).__name__}>"

    articles = _parse_news_html(html, ticker, limit)

    if not articles:
        return (
            f"<no Sina Finance news found for {ticker} "
            f"between {start_date} and {end_date}>"
        )

    news_str = f"## {ticker} 新闻 (新浪财经), {start_date} 至 {end_date}:\n\n"
    for a in articles:
        news_str += f"### {a['title']} (source: {a['source']})\n"
        news_str += f"Date: {a['date']}\n"
        news_str += f"Link: {a['link']}\n\n"
    return news_str
