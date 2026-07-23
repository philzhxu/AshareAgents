"""Sentiment analyst — multi-source sentiment analysis for a target ticker.

Previously named ``social_media_analyst``. Renamed and redesigned because
the old version had a prompt that demanded social-media analysis but the
only tool available was Yahoo Finance news — which led LLMs to fabricate
Reddit/X/StockTwits content under prompt pressure (verified live).

The redesigned agent pre-fetches two or more complementary data sources
before the LLM is invoked and injects them into the prompt as structured
blocks.  The source set depends on the ticker's market:

  **A-stock (``.SZ`` / ``.SS`` / ``.BJ`` suffix):**
    1. Sina Finance (新浪财经) — stock-specific news
    2. East Money Guba (东方财富股吧) — retail investor forum posts with
       engagement metrics (views, replies) and user nicknames
    3. Cninfo (巨潮资讯) — official investor interactive Q&A

  **US / other markets (default):**
    1. Yahoo Finance news — institutional framing
    2. StockTwits messages — retail-trader posts with Bullish/Bearish tags
    3. Reddit posts — r/wallstreetbets, r/stocks, r/investing

The agent does not use tool-calling; the data is in the prompt from
turn 0. Output uses the structured-output pattern (json_schema for
OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic), falling
back to free-text generation for providers that lack native support, so
the sentiment header (band + score + confidence) is deterministic across
runs and providers instead of free-form per-model prose.

See: https://github.com/TauricResearch/TradingAgents/issues/557
See: https://github.com/TauricResearch/TradingAgents/issues/796
"""

from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.schemas import SentimentReport, render_sentiment_report
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_news,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.cninfo import fetch_cninfo_qa
from tradingagents.dataflows.eastmoney_guba import fetch_eastmoney_posts
from tradingagents.dataflows.reddit import fetch_reddit_posts
from tradingagents.dataflows.sina_news import get_news_sina
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages
from tradingagents.dataflows.symbol_utils import is_ashare


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def create_sentiment_analyst(llm):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + community data, injects them into the prompt as
    structured blocks, and produces a deterministic sentiment report via
    structured output (with a free-text fallback for providers that do
    not support it).

    For A-stock tickers (``.SZ`` / ``.SS`` / ``.BJ``) the source set
    switches to Sina Finance news and East Money Guba forum posts.
    """
    structured_llm = bind_structured(llm, SentimentReport, "Sentiment Analyst")

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = get_instrument_context_from_state(state)

        # Choose source set based on market.
        if is_ashare(ticker):
            news_block = get_news_sina(ticker, start_date, end_date)
            guba_block = fetch_eastmoney_posts(ticker, limit=30)
            cninfo_block = fetch_cninfo_qa(ticker, limit=20)

            system_message = _build_system_message_cn(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                news_block=news_block,
                guba_block=guba_block,
                cninfo_block=cninfo_block,
            )
        else:
            # Pre-fetch all three sources. Each fetcher degrades gracefully and
            # returns a string (no exceptions surface from here), so the LLM
            # always sees something — either real data or a clear placeholder.
            news_block = get_news.func(ticker, start_date, end_date)
            stocktwits_block = fetch_stocktwits_messages(ticker, limit=30)
            reddit_block = fetch_reddit_posts(ticker)

            system_message = _build_system_message(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                news_block=news_block,
                stocktwits_block=stocktwits_block,
                reddit_block=reddit_block,
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}"
                    "\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # Format the template into a concrete message list so the structured
        # and free-text paths receive the same input. No bind_tools — the
        # data is already in the prompt.
        formatted_messages = prompt.format_messages(messages=state["messages"])

        report_text = invoke_structured_or_freetext(
            structured_llm,
            llm,
            formatted_messages,
            render_sentiment_report,
            "Sentiment Analyst",
        )

        return {
            "messages": [AIMessage(content=report_text)],
            "sentiment_report": report_text,
        }

    return sentiment_analyst_node


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
) -> str:
    """Assemble the US-market sentiment-analyst system message."""
    return f"""You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date}, drawing on three complementary data sources that have already been collected for you.

## Data sources (pre-fetched, in this prompt)

### News headlines — Yahoo Finance, past 7 days
Institutional framing. Fact-driven, slower-moving signal.

<start_of_news>
{news_block}
<end_of_news>

### StockTwits messages — retail-trader social platform indexed by cashtag
Fast-moving signal. Each message carries a user-labeled sentiment tag (Bullish / Bearish / no-label) plus the message body.

<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

### Reddit posts — r/wallstreetbets, r/stocks, r/investing (past 7 days)
Community discussion. Engagement signal via upvote score and comment count. Subreddit character matters (r/wallstreetbets is often contrarian/exuberant; r/stocks more measured; r/investing longer-term).

<start_of_reddit>
{reddit_block}
<end_of_reddit>

## How to analyze this data (best practices)

1. **Read the StockTwits Bullish/Bearish ratio as a leading retail-sentiment signal.** A 70/30 bullish/bearish split is moderately bullish; ≥90/10 may indicate over-extension and contrarian risk; 50/50 is uncertainty. Sample size matters — base rates on the actual message count, not percentages alone.

2. **Look for cross-source divergences.** If news framing is bearish but StockTwits is overwhelmingly bullish, that mismatch is itself a signal — it can mean retail is leaning into a thesis the news flow hasn't caught up to (or vice versa, that retail is chasing while institutions are cautious).

3. **Weight Reddit posts by engagement.** A 400-upvote / 200-comment thread reflects community attention; a 3-upvote post is noise. Read the body excerpts for context — the title alone often misleads.

4. **Distinguish opinion from event.** A news headline ("Nvidia announces $500M Corning deal") is an event; a StockTwits post ("buying NVDA, this is going to moon") is opinion. Both are inputs but should be weighted differently in your conclusions.

5. **Identify recurring narrative themes.** What topic keeps coming up across sources? That's the dominant narrative driving current sentiment.

6. **Be honest about data limits.** If StockTwits returned only a handful of messages, or one or more sources returned an "<unavailable>" placeholder, the sentiment read is less robust — flag this explicitly in the `confidence` field and the narrative. If the sources are silent on a given subreddit, say so.

7. **Identify catalysts and risks** that emerge across sources — news of upcoming earnings, product launches, competitive threats, macro headlines, etc.

8. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside fundamentals and technicals, not as a price call.

## Output fields

Fill the following fields:

- **overall_band**: Exactly one of Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. Use Mixed when sources point in clearly different directions; Neutral only when all sources are genuinely silent.
- **overall_score**: A number from 0 (maximally bearish) to 10 (maximally bullish); 5 is neutral. Keep it consistent with overall_band.
- **confidence**: low / medium / high, based on data quality and sample size.
- **narrative**: Full source-by-source breakdown, divergences, dominant narrative themes, catalysts and risks, and a markdown summary table of key sentiment signals (direction, source, supporting evidence).

{get_language_instruction()}"""


def _build_system_message_cn(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    guba_block: str,
    cninfo_block: str,
) -> str:
    """Assemble the A-stock (Chinese market) sentiment-analyst system message.

    Uses Sina Finance for institutional news, East Money Guba for retail
    forum discussion, and Cninfo for official investor relations Q&A —
    providing multi-angle coverage of Chinese A-stock market sentiment.
    """
    return f"""You are a financial market sentiment analyst specializing in the Chinese A-stock market. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date}, drawing on three complementary Chinese-market data sources that have already been collected for you.

## Data sources (pre-fetched, in this prompt)

### News headlines — 新浪财经 (Sina Finance), past 7 days
Institutional and media framing of the stock. Sina Finance is one of China's largest financial news portals, aggregating coverage from major Chinese financial media outlets. These articles represent the formal news narrative — slower-moving but authoritative.

<start_of_news>
{news_block}
<end_of_news>

### Forum posts — 东方财富股吧 (East Money Guba), most recent
East Money Guba is China's largest per-stock retail investor discussion forum — the closest equivalent to Reddit + StockTwits for A-stocks. Each post includes the user's nickname, post timestamp, view count (阅读), reply count (回复), and the full post body. High view counts signal community attention; high reply counts signal active debate. Pay attention to the language and emotional tone of the posts — Chinese retail investors often express sentiment through vivid metaphors, sarcasm, and memetic phrases.

<start_of_guba>
{guba_block}
<end_of_guba>

### Interactive Q&A — 巨潮资讯 (Cninfo), recent
Cninfo (巨潮资讯网) hosts the official interactive Q&A platform where investors ask questions directly to listed company management. The Q&A exchanges reveal what topics investors are most concerned about, management's communication style and transparency, and whether the company is addressing investor concerns substantively. Responsive, detailed answers signal good IR management; evasive or templated answers may signal issues. This is unique official-channel data — not sentiment speculation, but recorded interactions with the company itself.

<start_of_cninfo>
{cninfo_block}
<end_of_cninfo>

## How to analyze this data (best practices for Chinese markets)

1. **Read the Guba post sentiment through the lens of Chinese retail investor psychology.** Chinese retail investors (散户) dominate A-stock trading volume. Look for:
   - **Bullish signals**: optimism about policy support (政策利好), bottom-fishing language (抄底), expectations of institutional buying (国家队入场, 北向资金), excitement about a sector rotation (板块轮动).
   - **Bearish signals**: complaints about price manipulation (割韭菜), fear of delisting/ST risk, frustration with continuous decline (阴跌, 跌跌不休), comparison to stronger-performing sectors.
   - **Contrarian indicators**: Extreme bearishness in Guba posts when the stock is already deeply sold off can be a contrarian bottom signal; universal bullishness after a sharp rally can signal a top.

2. **Use Cninfo Q&A to ground sentiment in official information.** Investor questions to management reveal what the market is worried about; management's answers reveal how the company is positioning itself. If management dodges a specific question (e.g. about declining margins or regulatory issues), that avoidance is itself a sentiment signal. If Q&A volume spikes around a specific topic, that topic is the current narrative driver.

3. **Look for cross-source divergences.** Key patterns to watch for:
   - Sina Finance neutral but Guba emotional → market is moving on sentiment, not news
   - Guba bullish but Sina Finance cautious → retail enthusiasm may be ahead of fundamentals
   - Cninfo Q&A focused on a specific risk topic that news/Guba don't mention yet → emerging concern
   - All three sources pointing in the same direction → strong consensus signal

4. **Weight posts by engagement metrics.** A post with 50,000+ views and 30+ replies carries far more community signal than a post with single-digit views. High-engagement posts often set the narrative tone for the entire forum.

5. **Distinguish news events from retail chatter.** A Sina Finance article about actual company developments (earnings, regulatory filings, major contracts) is an event. A Guba post saying "白酒yyds,冲!" (baijiu forever, charge!) is pure sentiment. Weight events higher for fundamental assessment; weight chatter higher for short-term sentiment temperature.

6. **Identify recurring narrative themes.** What topics keep appearing across all three sources? Common A-stock narrative themes include: policy direction (政策方向), sector rotation (板块轮动), earnings surprises, margin trading activity (融资融券), north-bound capital flows (北向资金), and macro-economic indicators.

7. **Be honest about data limits.** If one or more sources returned an "<unavailable>" placeholder or only a handful of posts, the sentiment read is less robust — flag this explicitly in the `confidence` field and narrative.

8. **Past sentiment is not predictive.** Frame your conclusions as a temperature check for the trader to weigh alongside fundamentals and technicals, not as a price forecast.

## Output fields

Fill the following fields:

- **overall_band**: Exactly one of Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. Use Mixed when sources point in clearly different directions; Neutral only when all sources are genuinely silent.
- **overall_score**: A number from 0 (maximally bearish) to 10 (maximally bullish); 5 is neutral. Keep it consistent with overall_band.
- **confidence**: low / medium / high, based on data quality and sample size.
- **narrative**: Full source-by-source breakdown, divergences, dominant narrative themes, catalysts and risks, and a markdown summary table of key sentiment signals (direction, source, supporting evidence).

{get_language_instruction()}"""


# ---------------------------------------------------------------------------
# Backwards-compatibility shim
# ---------------------------------------------------------------------------
def create_social_media_analyst(llm):
    """Deprecated alias for :func:`create_sentiment_analyst`.

    Kept so existing code that imports ``create_social_media_analyst``
    continues to work.

    .. deprecated::
        Import :func:`create_sentiment_analyst` directly instead.
    """
    import warnings
    warnings.warn(
        "create_social_media_analyst is deprecated and will be removed in a "
        "future version. Use create_sentiment_analyst instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_sentiment_analyst(llm)
