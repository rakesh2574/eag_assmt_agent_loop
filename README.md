# EAG V3 — Session 6: Agentic Architecture (Perception / Memory / Decision / Action)

The **agent is the Python orchestration loop** in `agent6.py` — not the LLM. The
LLMs only act out roles. The loop moves all data between roles, attaches all
context, tracks goal state, and decides termination. Every cross-role boundary
is a Pydantic v2 model (`schemas.py`); there is no free-form dict passing and no
regex parsing of LLM output. LLMs are stateless, so the loop rebuilds context
from scratch every iteration (memory re-read, history + artifact refs re-fed).

## The four cognitive roles

| Module | Role | LLM use |
|---|---|---|
| `memory.py` | Typed service beside the loop. Facts, preferences, tool outcomes, scratchpad. | Reads are pure keyword search (**no LLM**). Only `remember()` makes **one classify call**, pinned to Gemini. Persists non-scratchpad to `state/memory.json`. |
| `perception.py` | Orchestrator/CEO **and verifier**. Decomposes on run 1, then only flips `done` flags / sets attachments. | **One** call, forced to Gemini (`provider="g"`, `temperature=1.0`). |
| `decision.py` | Worker. One goal at a time, never sees the others. Returns answer **or** one tool call. | **One** call, `auto_route="decision"`. |
| `action.py` | Pure dispatch (~30 lines). Awaits the MCP tool, flattens to text, text-vs-artifact at 4 KB. | **No LLM.** |

Support pieces: `artifacts.py` (content-addressable `ArtifactStore`), `gateway.py`
(the only door to an LLM — talks exclusively to LLM Gateway V3), `agent6.py` (the
loop). The MCP server (`mcp_server.py`, nine tools over stdio) and `llm_gatewayV3/`
are used **unchanged**.

## Anti-hallucination design (enforced in Python, not by trusting the model)

* **Positional goal identity** — Perception's wire schema (`PerceptionOutput`) has
  **no goal-id field**. Goals are identified by position; the loop carries prior
  ids and maps them to new positions. The model has nowhere to invent a stale id.
* **Integer artifact indices** — artifact-bearing memory hits are presented with an
  integer index. Perception emits `artifact_index: <int>`; the loop maps it back to
  the real `art:` handle.
* **Attachment existence guard** — before attaching, `artifacts.exists(handle)` is
  checked. A hallucinated handle/index is silently dropped.
* **Force-attach safety net** — for a synthesis goal (text contains synthesise /
  extract / list / compare / decide …) with an artifact in the hits, the most recent
  artifact is auto-attached.
* **Substantive-answer rule** — Decision's prompt forbids meta-answers ("the page is
  fetched, how shall I proceed?") for extraction/list/compare/select goals.

## Prerequisites

1. **LLM Gateway V3 running** on `http://localhost:8101` (the substrate for *every*
   LLM call). Start it: `cd llm_gatewayV3 && ./run.sh`. Verify:
   `curl -s http://localhost:8101/v1/routers`. It needs at least a Gemini key in its
   `.env` (Perception + Memory are pinned to Gemini).
2. **`uv`** installed (all deps and execution go through it).
3. **`.env`** in this folder: `cp .env.example .env` and set `TAVILY_API_KEY`
   (optional but recommended — without it `web_search` falls back to DuckDuckGo).
4. First `uv run` will install `crawl4ai`; you may also need its browser once:
   `uv run python -m playwright install chromium`.

## Running the four acceptance queries

A fresh attempt starts from a clean `state/`:

```bash
rm -rf state/        # clean slate
```

```bash
# A — Shannon (artifact attach, ~3 iters)
uv run python agent6.py "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory."

# B — Tokyo (multi-goal + carryover, ~6 iters)
uv run python agent6.py "Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate."

# C — Mom's birthday (durable memory across TWO runs, ~4 + ~2 iters)
#   Run 1 writes the fact; Run 2 reads it back from the SAME state/.
uv run python agent6.py "My mom's birthday is 15 May 2026. Remember that and give me a calendar reminder for two weeks before and on the day."
uv run python agent6.py "When is mom's birthday?"

# D — asyncio (multi-source synthesis, ~5-7 iters)
uv run python agent6.py "Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on."
```

`MAX_ITERATIONS` defaults to 12; override per-run with `--max-iters N` or via the
`MAX_ITERATIONS` env var. Each run prints an iteration trace (memory hits, the
goal list with `done` flags and attachments, the Decision move, the Action result)
and a final answer block.

> **Note on large attachments.** A fetched page (e.g. Shannon ≈ 250 KB) exceeds the
> router's HUGE ceiling, so for a Decision step with a big attachment the loop uses
> the gateway's documented escape hatch (`provider="g"`, Gemini long context) and
> caps the attached window to 48 KB. Every call still goes through Gateway V3.

---

## Captured terminal output (clean `state/`) — PASTE REAL OUTPUT BELOW

> These blocks are placeholders. Run each query from a clean `state/` on the machine
> where the gateway is up, then paste the actual terminal traces here. (The build
> environment used to author this code cannot reach `localhost:8101` or run headless
> Chromium, so these were intentionally left for you to capture.)

### A — Shannon

```text
<paste terminal output of query A here>
```

### B — Tokyo

```text
<paste terminal output of query B here>
```

### C — Mom's birthday (Run 1 then Run 2, same state/)

```text
<paste terminal output of query C run 1 here>

<paste terminal output of query C run 2 here>
```

### D — asyncio

```text
<paste terminal output of query D here>
```

---

## Files

```
agent6.py        the loop (the agent)
schemas.py       Pydantic v2 boundary contracts
memory.py        typed memory service (keyword reads, one classify write)
perception.py    orchestrator + verifier (Gemini, temp 1.0)
decision.py      worker (auto_route="decision")
action.py        pure dispatch, no LLM
artifacts.py     content-addressable artifact store
gateway.py       the only door to an LLM (-> Gateway V3)
mcp_server.py    provided, unchanged — 9 tools over stdio
llm_gatewayV3/   provided, unchanged — the LLM substrate
prompts/         perception.txt, decision.txt, pop_validation.json
state/           runtime state (gitignored; rm -rf to reset)
```

See `RUN.md` for the exact command sequence.
