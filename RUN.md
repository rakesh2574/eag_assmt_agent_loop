# RUN.md — exact command sequence

All commands run from this folder (the one containing `agent6.py`).

## 0. One-time setup

```bash
# (a) Tell the agent loop about Tavily (optional but recommended).
cp .env.example .env
# edit .env and set TAVILY_API_KEY=...   (leave blank to use DuckDuckGo fallback)

# (b) Make sure the LLM Gateway V3 has provider keys (at least Gemini).
#     The gateway reads its OWN .env (../.env relative to llm_gatewayV3/, i.e.
#     this folder's .env works if it contains GEMINI_API_KEY etc).
#     See llm_gatewayV3/README.md for the full key list.
```

## 1. Start the LLM Gateway V3 (separate terminal — leave it running)

```bash
cd llm_gatewayV3
./run.sh                       # first run creates .venv, then serves on :8101
# verify in another shell:
curl -s http://localhost:8101/v1/routers | python3 -m json.tool
```

If that curl fails, the gateway is not up — start it before running the agent.

## 2. (first time only) prime crawl4ai's browser

```bash
uv run python -m playwright install chromium
```

## 3. Clean state between attempts

```bash
rm -rf state/                  # a fresh start = delete state/
```

## 4. Run each query

```bash
# A — Shannon (~3 iters)
uv run python agent6.py "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory."

# B — Tokyo (~6 iters)
uv run python agent6.py "Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate."

# C — Mom's birthday — TWO runs against the SAME state/ (do NOT rm between them)
uv run python agent6.py "My mom's birthday is 15 May 2026. Remember that and give me a calendar reminder for two weeks before and on the day."
uv run python agent6.py "When is mom's birthday?"

# D — asyncio (~5-7 iters)
uv run python agent6.py "Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on."
```

Override the iteration budget if needed:

```bash
uv run python agent6.py "<query>" --max-iters 8
# or:  MAX_ITERATIONS=8 uv run python agent6.py "<query>"
```

## Notes

* **Query C is the persistence test:** run 1 and run 2 share `state/memory.json`.
  Run `rm -rf state/` *before* run 1, but **not** between run 1 and run 2.
* Each run prints a full iteration trace plus a `FINAL ANSWER:` block.
* `state/` is gitignored and safe to delete at any time.
