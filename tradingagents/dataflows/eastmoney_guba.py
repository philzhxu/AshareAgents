"""East Money Guba (东方财富股吧) forum post fetcher for A-stock tickers.

East Money (东方财富) is China's largest financial portal, and its "Guba"
(股吧, stock bar/forum) is the most active per-stock discussion board in the
Chinese market — the closest equivalent to a Reddit + StockTwits hybrid for
A-stocks. Each stock has its own dedicated forum page at
``guba.eastmoney.com/list,{code}.html``.

The HTML page embeds a JavaScript ``article_list`` variable containing the
most recent posts in JSON format, which this module extracts and reformats
into a plaintext block suitable for prompt injection.

No API key required.  Uses ``curl_cffi`` for TLS fingerprint impersonation
(Chinese financial sites block standard Python HTTP clients).  Degrades
gracefully — returns a placeholder string rather than raising, so callers
never special-case missing data.
"""

from __future__ import annotations

import html as _html
import json
import logging
import re
import time
from datetime import datetime

from .symbol_utils import ashare_bare_code

logger = logging.getLogger(__name__)

_GUBA_URL = "https://guba.eastmoney.com/list,{code}.html"
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _parse_article_list_json(html: str) -> list[dict] | None:
    """Extract the ``article_list`` JSON from an East Money Guba HTML page.

    The embedded data is a large JSON object (≈90 KB) assigned to a JS
    variable.  We locate it by scanning for ``article_list=`` and then
    brace-count to find the matching closing ``}`` — this handles nesting
    correctly, whereas a regex lazy-match stops at the first ``}`` (which
    belongs to a nested object, not the top-level one).
    """
    # Locate the start of the JSON payload.
    start_marker = "article_list="
    idx = html.find(start_marker)
    if idx < 0:
        return None

    json_start = html.find("{", idx)
    if json_start < 0:
        return None

    # Brace-count to find the matching closing brace.
    depth = 0
    in_string = False
    escape = False
    json_end = json_start
    max_pos = min(json_start + 500_000, len(html))
    for i in range(json_start, max_pos):
        c = html[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                json_end = i + 1
                break

    raw = html[json_start:json_end]
    if not raw:
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("Failed to parse Guba article_list JSON (%d bytes)", len(raw))
        return None

    return data.get("re", []) if isinstance(data, dict) else None


def _strip_html_tags(text: str) -> str:
    """Remove HTML tags and decode entities from *text*."""
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = _html.unescape(cleaned)
    return " ".join(cleaned.split())


def _parse_bullish_bearish(post: dict) -> str | None:
    """Return a sentiment label from a Guba post's ``bullish_bearish`` field."""
    val = post.get("bullish_bearish")
    if val == 1 or val == "1":
        return "看涨"  # bullish
    elif val == 2 or val == "2":
        return "看跌"  # bearish
    return None


def fetch_eastmoney_posts(
    ticker: str,
    limit: int = 30,
    timeout: float = 15.0,
) -> str:
    """Fetch recent East Money Guba forum posts for *ticker*.

    Args:
        ticker: A-stock ticker (e.g. ``000858.SZ``, ``601318.SS``).
        limit: Maximum number of posts to return.
        timeout: HTTP request timeout in seconds.

    Returns:
        Formatted plaintext block with per-post entries, or a placeholder
        string when the source is unreachable.
    """
    code = ashare_bare_code(ticker) or ticker.strip().upper()
    url = _GUBA_URL.format(code=code)

    try:
        from curl_cffi import requests as cffi_requests
    except ImportError:
        logger.warning("curl_cffi not available — falling back to urllib for Guba")
        return _fetch_eastmoney_posts_urllib(ticker, code, url, limit, timeout)

    try:
        resp = cffi_requests.get(
            url,
            impersonate="chrome120",
            timeout=timeout,
            headers={
                "User-Agent": _UA,
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept": "text/html,application/xhtml+xml",
                "Referer": "https://guba.eastmoney.com/",
            },
        )
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:
        logger.warning("East Money Guba fetch failed for %s: %s", ticker, exc)
        return f"<东方财富股吧不可用: {type(exc).__name__}>"

    posts = _parse_article_list_json(html)

    if not posts:
        # Fallback: try to extract post titles and content from HTML body text
        posts = _fallback_parse_html(html)
        if not posts:
            return f"<no East Money Guba posts found for {ticker}>"

    return _format_posts(posts[:limit], ticker)


def _fetch_eastmoney_posts_urllib(
    ticker: str,
    code: str,
    url: str,
    limit: int,
    timeout: float,
) -> str:
    """Fallback using standard-library ``urllib`` when ``curl_cffi`` is absent."""
    import http.client
    from urllib.request import Request, urlopen

    req = Request(
        url,
        headers={
            "User-Agent": _UA,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning("East Money Guba (urllib) fetch failed for %s: %s", ticker, exc)
        return f"<东方财富股吧不可用: {type(exc).__name__}>"

    posts = _parse_article_list_json(html)
    if not posts:
        posts = _fallback_parse_html(html)
        if not posts:
            return f"<no East Money Guba posts found for {ticker}>"

    return _format_posts(posts[:limit], ticker)


def _fallback_parse_html(html: str) -> list[dict]:
    """Crude fallback: extract visible post-like content from the page body."""
    # Remove scripts, styles, and the footer
    body = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    body = re.sub(r"<style[^>]*>.*?</style>", "", body, flags=re.DOTALL)
    # Extract text content between known post markers
    text = re.sub(r"<[^>]+>", "\n", body)
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 15]
    if not lines:
        return []
    # Return as pseudo-posts
    posts = []
    for line in lines[:30]:
        posts.append({
            "post_title": line[:120],
            "post_content": line[:280],
            "post_publish_time": "?",
            "post_user": {"user_nickname": "?"},
            "post_click_count": 0,
            "post_comment_count": 0,
            "bullish_bearish": 0,
        })
    return posts


def _format_posts(posts: list[dict], ticker: str) -> str:
    """Format Guba posts into a plaintext block for prompt injection."""
    total = len(posts)
    bullish = sum(
        1 for p in posts if _parse_bullish_bearish(p) == "看涨"
    )
    bearish = sum(
        1 for p in posts if _parse_bullish_bearish(p) == "看跌"
    )
    neutral = total - bullish - bearish

    bull_pct = round(100 * bullish / total) if total else 0
    bear_pct = round(100 * bearish / total) if total else 0

    summary = (
        f"看涨 (Bullish): {bullish} ({bull_pct}%) · "
        f"看跌 (Bearish): {bearish} ({bear_pct}%) · "
        f"中性 (Neutral): {neutral} · "
        f"Total: {total} recent posts"
    )

    lines = [
        f"东方财富股吧 (East Money Guba) — {ticker} 论坛帖子",
        f"Sentiment summary: {summary}",
        "",
    ]

    for p in posts:
        title = (p.get("post_title") or "").replace("\n", " ").strip()
        content = _strip_html_tags(p.get("post_content") or "")
        pub_time = p.get("post_publish_time", "?")

        # User nickname: most posts store it at the top level
        # (``user_nickname``), but a few featured/repost entries nest
        # it inside a ``post_user`` dict.
        user = "?"
        if isinstance(p.get("post_user"), dict):
            user = p["post_user"].get("user_nickname", "?")
        if user in (None, "?"):
            user = p.get("user_nickname", "?")

        clicks = p.get("post_click_count", 0)
        comments = p.get("post_comment_count", 0)
        sentiment = _parse_bullish_bearish(p)

        meta = f"{pub_time} · @{user} · {clicks}阅读 · {comments}回复"
        if sentiment:
            meta += f" · [{sentiment}]"

        if len(content) > 280:
            content = content[:280] + "…"

        lines.append(f"[{meta}] {title}")
        if content and content != title:
            lines.append(f"    {content}")

    return "\n".join(lines)
