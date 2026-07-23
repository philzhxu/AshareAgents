"""SSE (上海证券交易所) disclosure announcement fetcher for A-stock tickers.

The Shanghai Stock Exchange (上交所) publishes company announcements/disclosures
at ``www.sse.com.cn/disclosure/listedinfo/summaries/``.  This module fetches
recent announcements for a given SSE-listed ticker (``.SS`` / ``.SH`` suffix).

The SSE website uses a legacy query API at ``query.sse.com.cn`` with JSONP
callbacks plus a newer static CDN at ``static.sse.com.cn`` for full-text PDF
files.  This module tries the query API first and falls back to HTML scraping
of the disclosure summary page.  Like every data source in the framework, it
degrades gracefully — returns a placeholder string rather than raising.

Note: This module only covers Shanghai-listed stocks (``.SS`` / ``.SH``).
Shenzhen-listed stocks (``.SZ``) should use the Cninfo interactive Q&A module
or Sina Finance news.
"""

from __future__ import annotations

import html as _html
import json
import logging
import re

from .symbol_utils import ashare_bare_code

logger = logging.getLogger(__name__)

# SSE query API endpoint (legacy, may return empty for newer listings).
_QUERY_URL = "https://query.sse.com.cn/security/stock/queryCompanyBulletin.do"

# SSE disclosure summary page — lists recent announcements.
_DISCLOSURE_URL = "https://www.sse.com.cn/disclosure/listedinfo/summaries/"

# Alternative: the announcement search page.
_ANNOUNCEMENT_URL = "https://www.sse.com.cn/disclosure/listedinfo/announcement/"

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _strip_html_tags(text: str) -> str:
    """Remove HTML tags and decode entities from *text*."""
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = _html.unescape(cleaned)
    return " ".join(cleaned.split())


def _parse_jsonp(text: str) -> dict | None:
    """Parse a JSONP response into a dict. Returns None on failure."""
    if not text:
        return None
    # Strip the JSONP wrapper: ``jsonpCallback({...})`` → ``{...}``
    m = re.match(r"^\w+?\((.*)\);?\s*$", text.strip(), re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _parse_query_api(data: dict, limit: int) -> list[dict]:
    """Extract announcement items from SSE query API JSON."""
    items = []
    page_help = data.get("pageHelp") or data
    results = page_help.get("data") or data.get("result") or []
    if not isinstance(results, list):
        return items
    for item in results:
        if not isinstance(item, dict):
            continue
        title = (item.get("TITLE") or item.get("title") or item.get("BULLETIN_TITLE") or "").strip()
        if not title:
            continue
        items.append({
            "title": title,
            "date": str(item.get("START_DATE") or item.get("SSEDATE") or item.get("date") or item.get("bulletinDate") or "?"),
            "link": str(item.get("URL") or item.get("bulletinUrl") or ""),
            "company": str(item.get("SECURITY_CODE") or item.get("COMPANY_ABBR") or ""),
        })
        if len(items) >= limit:
            break
    return items


def _parse_html_announcements(html_text: str, limit: int) -> list[dict]:
    """Extract announcement items from an SSE disclosure HTML page."""
    items = []

    # Pattern 1: standard announcement table rows.
    # SSE pages often use <tr> with nested <td> for date, company, title.
    tr_blocks = re.findall(
        r"<tr[^>]*>(.*?)</tr>", html_text, re.DOTALL,
    )
    for tr in tr_blocks:
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)
        if len(tds) < 2:
            continue
        # Try to extract: date, company code/name, title, link
        date_str = ""
        title = ""
        link = ""
        for td in tds:
            td_text = _strip_html_tags(td)
            # Date pattern: YYYY-MM-DD or YYYY年MM月DD日
            if re.match(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}", td_text):
                date_str = td_text
            elif len(td_text) > 10 and not re.match(r"^\d{6}$", td_text):
                title = td_text
                # Extract link
                link_match = re.search(r'href=[\"\\x27]([^\"\\x27]+)[\"\\x27]', td)
                if link_match:
                    link = link_match.group(1)
                    if link.startswith("/"):
                        link = "https://www.sse.com.cn" + link
        if title:
            items.append({
                "title": title[:200],
                "date": date_str,
                "link": link,
                "company": "",
            })
            if len(items) >= limit:
                break

    # Pattern 2: list items with announcement links.
    if not items:
        link_matches = re.findall(
            r'<a[^>]*href=[\"\\x27]([^\"\\x27]*(?:announcement|disclosure|bulletin|c)[^\"\\x27]*\.(?:shtml|html|pdf))[\"\\x27][^>]*>([^<]{10,200})</a>',
            html_text,
        )
        for link, title in link_matches:
            title = title.strip()
            if len(title) > 10:
                if link.startswith("/"):
                    link = "https://www.sse.com.cn" + link
                items.append({
                    "title": title[:200],
                    "date": "?",
                    "link": link,
                    "company": "",
                })
                if len(items) >= limit:
                    break

    return items


def fetch_sse_announcements(
    ticker: str,
    limit: int = 15,
    timeout: float = 15.0,
) -> str:
    """Fetch recent SSE announcements for *ticker*.

    Only covers Shanghai-listed stocks (``.SS`` / ``.SH`` suffix).  Shenzhen
    stocks should use other data sources (Cninfo, Sina).

    Args:
        ticker: A-stock ticker (e.g. ``600519.SS``, ``601318.SH``).
        limit: Maximum number of announcements to return.
        timeout: HTTP request timeout in seconds.

    Returns:
        Formatted plaintext block suitable for prompt injection, or a
        placeholder string when the source is unreachable.
    """
    code = ashare_bare_code(ticker) or ticker.strip().upper()

    try:
        from curl_cffi import requests as cffi_requests
    except ImportError:
        logger.warning("curl_cffi not available — falling back to urllib for SSE")
        return _fetch_sse_urllib(ticker, code, limit, timeout)

    headers = {
        "User-Agent": _UA,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://www.sse.com.cn/disclosure/listedinfo/summaries/",
    }

    items = []

    # Attempt 1: Query API with JSONP.
    try:
        resp = cffi_requests.get(
            _QUERY_URL,
            impersonate="chrome120",
            timeout=timeout,
            headers=headers,
            params={
                "jsonCallBack": "jsonpCallback",
                "stockCode": code,
                "pageSize": limit,
                "pageNo": 1,
                "isPagination": "true",
            },
        )
        resp.raise_for_status()
        data = _parse_jsonp(resp.text)
        if data:
            items = _parse_query_api(data, limit)
    except Exception as exc:
        logger.debug("SSE query API failed for %s: %s", ticker, exc)

    # Attempt 2: HTML scraping of the disclosure summary page.
    if not items:
        try:
            html_headers = {**headers, "Accept": "text/html,application/xhtml+xml"}
            resp = cffi_requests.get(
                _DISCLOSURE_URL,
                impersonate="chrome120",
                timeout=timeout,
                headers=html_headers,
            )
            resp.raise_for_status()
            items = _parse_html_announcements(resp.text, limit)
        except Exception as exc:
            logger.warning("SSE HTML scraping failed for %s: %s", ticker, exc)

    if not items:
        return f"<no SSE announcements found for {ticker}>"

    return _format_announcements(items, ticker)


def _fetch_sse_urllib(
    ticker: str,
    code: str,
    limit: int,
    timeout: float,
) -> str:
    """Fallback using standard-library ``urllib``."""
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    headers_dict = {
        "User-Agent": _UA,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    items = []

    # Attempt 1: Query API.
    params = urlencode({
        "jsonCallBack": "jsonpCallback",
        "stockCode": code,
        "pageSize": limit,
        "pageNo": 1,
    })
    try:
        req = Request(f"{_QUERY_URL}?{params}", headers={
            **headers_dict, "Accept": "application/json",
            "Referer": "https://www.sse.com.cn/",
        })
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = _parse_jsonp(raw)
        if data:
            items = _parse_query_api(data, limit)
    except Exception as exc:
        logger.debug("SSE query API (urllib) failed: %s", exc)

    # Attempt 2: HTML scraping.
    if not items:
        try:
            req = Request(_DISCLOSURE_URL, headers={
                **headers_dict, "Accept": "text/html",
            })
            with urlopen(req, timeout=timeout) as resp:
                html_text = resp.read().decode("utf-8", errors="replace")
            items = _parse_html_announcements(html_text, limit)
        except Exception as exc:
            logger.warning("SSE (urllib) HTML scraping failed: %s", exc)

    if not items:
        return f"<no SSE announcements found for {ticker}>"

    return _format_announcements(items, ticker)


def _format_announcements(items: list[dict], ticker: str) -> str:
    """Format SSE announcements into a plaintext block for prompt injection."""
    lines = [
        f"上交所公告 (SSE Announcements) — {ticker}",
        f"Total: {len(items)} recent announcements",
        "",
    ]

    for item in items:
        lines.append(f"- [{item['date']}] {item['title']}")
        if item.get("link"):
            lines.append(f"  链接: {item['link']}")

    return "\n".join(lines)
