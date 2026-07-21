from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

# DEFAULT_CONFIG already applies TRADINGAGENTS_* env-var overrides
# (llm_provider, deep_think_llm, quick_think_llm, backend_url, etc.),
# so users can switch models or endpoints purely via .env without
# editing this script. Override individual keys here only when you
# want a hard-coded value that should ignore the environment.
config = DEFAULT_CONFIG.copy()

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# A-stock example (AShareAgents auto-detects .SZ/.SS/.BJ tickers and
# switches to Chinese data sources: East Money, Sina Finance, Guba).
# For US stocks use tickers like "AAPL" or "NVDA" (no suffix) — those
# will use Yahoo Finance, Reddit, and StockTwits as in the original
# TradingAgents framework.
_, decision = ta.propagate("000858.SZ", "2026-07-21")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
