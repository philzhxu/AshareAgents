"""Cninfo (巨潮资讯) interactive Q&A fetcher for A-stock tickers.

Cninfo (巨潮资讯网) is the official designated information disclosure platform
for Chinese listed companies, operated by Shenzhen Securities Information Co.,
Ltd.  Its "互动易" (Interactive Easy) platform at ``irm.cninfo.com.cn`` hosts
interactive Q&A between investors and listed company management — the closest
equivalent to an official investor relations Q&A feed for A-stocks.

The platform is a Single-Page Application (SPA).  This module attempts to
access the underlying JSON API endpoints with ``curl_cffi`` TLS fingerprint
impersonation.  When the API is unreachable the module degrades gracefully
with a placeholder string, like every other data source in the framework.

Note: The interactive Q&A platform is operated by SZSE (深交所) and primarily
covers Shenzhen-listed stocks (``.SZ``).  Shanghai-listed stocks (``.SS``)
may have thinner coverage on this platform — their official IR Q&A is hosted
on the SSE e-interaction platform (``sseinfo.com``).
"""

from __future__ import annotations

import html as _html
import json
import logging
import re

from .symbol_utils import ashare_bare_code

logger = logging.getLogger(__name__)

# Cninfo Interactive Q&A API endpoints (community-documented, subject to change).
# The platform uses a nginx reverse proxy; POST requests are blocked (405) so
# we use GET with query parameters.
_QA_API = "https://irm.cninfo.com.cn/ircs/interactiveAnswer/getInteractiveAnswerList"

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


def _parse_api_response(data: dict, limit: int) -> list[dict]:
    """Extract Q&A items from a Cninfo API JSON response.

    The response structure varies; this function tries several common shapes.
    """
    items = []

    # Try common data paths
    candidates = []
    if isinstance(data, dict):
        candidates = [
            data.get("list"),
            data.get("data"),
            data.get("result"),
            data.get("rows"),
            data.get("questions"),
            (data.get("data") or {}).get("list") if isinstance(data.get("data"), dict) else None,
            (data.get("result") or {}).get("list") if isinstance(data.get("result"), dict) else None,
        ]

    for candidate in candidates:
        if isinstance(candidate, list) and candidate:
            for item in candidate:
                if not isinstance(item, dict):
                    continue

                question = (
                    item.get("questionContent")
                    or item.get("question")
                    or item.get("content")
                    or item.get("title")
                    or ""
                ).strip()
                answer = (
                    item.get("answerContent")
                    or item.get("replyContent")
                    or item.get("answer")
                    or item.get("reply")
                    or ""
                ).strip()

                if not question and not answer:
                    continue

                items.append({
                    "question": _strip_html_tags(question)[:500],
                    "answer": _strip_html_tags(answer)[:500],
                    "ask_time": str(item.get("questionTime") or item.get("askTime") or item.get("createTime") or "?"),
                    "reply_time": str(item.get("answerTime") or item.get("replyTime") or item.get("updateTime") or "?"),
                    "asker": str(item.get("questioner") or item.get("asker") or item.get("nickName") or "投资者"),
                    "status": str(item.get("replyStatus") or item.get("status") or "?"),
                })
                if len(items) >= limit:
                    break
            if items:
                break

    return items


def fetch_cninfo_qa(
    ticker: str,
    limit: int = 20,
    timeout: float = 15.0,
) -> str:
    """Fetch recent interactive Q&A entries for *ticker* from Cninfo.

    Args:
        ticker: A-stock ticker (e.g. ``000858.SZ``, ``601318.SS``).
        limit: Maximum number of Q&A entries to return.
        timeout: HTTP request timeout in seconds.

    Returns:
        Formatted plaintext block suitable for prompt injection, or a
        placeholder string when the source is unreachable.
    """
    code = ashare_bare_code(ticker) or ticker.strip().upper()

    try:
        from curl_cffi import requests as cffi_requests
    except ImportError:
        logger.warning("curl_cffi not available — cannot fetch Cninfo Q&A")
        return _fetch_cninfo_urllib(ticker, code, limit, timeout)

    headers = {
        "User-Agent": _UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://irm.cninfo.com.cn/views/interactiveAnswer",
    }

    try:
        resp = cffi_requests.get(
            _QA_API,
            impersonate="chrome120",
            timeout=timeout,
            headers=headers,
            params={
                "stockCode": code,
                "pageNum": 1,
                "pageSize": min(limit, 50),
            },
        )
        resp.raise_for_status()
        raw = resp.text
    except Exception as exc:
        logger.warning("Cninfo Q&A fetch failed for %s: %s", ticker, exc)
        return f"<巨潮资讯互动问答不可用: {type(exc).__name__}>"

    # The API may return JSON or an HTML SPA page.
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # The endpoint returned HTML (probably the SPA), not JSON.
        # Try to extract any embedded data.
        return _parse_html_fallback(raw, ticker, code, limit)

    items = _parse_api_response(data, limit)

    if not items:
        return f"<no Cninfo Q&A entries found for {ticker}>"

    return _format_qa(items, ticker)


def _fetch_cninfo_urllib(
    ticker: str,
    code: str,
    limit: int,
    timeout: float,
) -> str:
    """Fallback using standard-library ``urllib``."""
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    params = urlencode({"stockCode": code, "pageNum": 1, "pageSize": limit})
    url = f"{_QA_API}?{params}"
    req = Request(url, headers={
        "User-Agent": _UA,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept": "application/json",
    })
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Cninfo Q&A (urllib) fetch failed for %s: %s", ticker, exc)
        return f"<巨潮资讯互动问答不可用: {type(exc).__name__}>"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _parse_html_fallback(raw, ticker, code, limit)

    items = _parse_api_response(data, limit)
    if not items:
        return f"<no Cninfo Q&A entries found for {ticker}>"
    return _format_qa(items, ticker)


def _parse_html_fallback(html_text: str, ticker: str, code: str, limit: int) -> str:
    """Crude fallback: extract visible Q&A-like text from an SPA HTML page."""
    body = re.sub(r"<script[^>]*>.*?</script>", "", html_text, flags=re.DOTALL)
    body = re.sub(r"<style[^>]*>.*?</style>", "", body, flags=re.DOTALL)
    body = re.sub(r"<[^>]+>", "\n", body)
    lines = [line.strip() for line in body.split("\n") if len(line.strip()) > 15]

    if not lines:
        return f"<no Cninfo Q&A entries found for {ticker}>"

    # Return whatever readable text we could extract.
    qa_text = "\n".join(f"- {line}" for line in lines[:limit])
    return (
        f"## {ticker} 互动问答 (巨潮资讯)\n\n"
        f"(从网页提取的文本，可能包含非问答内容)\n\n{qa_text}"
    )


def _format_qa(items: list[dict], ticker: str) -> str:
    """Format Cninfo Q&A items into a plaintext block for prompt injection."""
    lines = [
        f"巨潮资讯互动问答 (Cninfo Interactive Q&A) — {ticker}",
        f"Total: {len(items)} recent Q&A exchanges",
        "",
    ]

    for item in items:
        lines.append(f"**问** ({item['asker']} · {item['ask_time']}):")
        lines.append(f"    {item['question']}")
        if item["answer"]:
            status = f" [{item['status']}]" if item["status"] != "?" else ""
            lines.append(f"**答** ({item['reply_time']}{status}):")
            lines.append(f"    {item['answer']}")
        else:
            lines.append(f"**答**: (尚未回复)")
        lines.append("")

    return "\n".join(lines)
