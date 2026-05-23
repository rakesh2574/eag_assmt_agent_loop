"""
gateway.py — the ONLY door to an LLM in this project.

Every LLM call (Perception, Memory classify, Decision) goes through here, and
this module talks exclusively to LLM Gateway V3 (http://localhost:8101) using
the vendored client in `llm_gatewayV3/client.py`. No provider SDK is imported
anywhere in the agent. No direct HTTP to providers.

It exposes one helper, `structured()`, which:
  * sends a system prompt + a serialized state prompt,
  * attaches a Pydantic model's JSON Schema via `response_format`,
  * returns the parsed dict (preferring the gateway's own `parsed`, falling back
    to JSON-decoding the text — JSON deserialization, NOT regex parsing of
    semantic content), plus the raw response for trace/debug.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Type

import httpx
from pydantic import BaseModel

# The worker pinned for Perception + Memory (and used as Decision's fallback).
# Default "g" = Gemini. Set AGENT_PROVIDER=gh in .env to use GitHub Models'
# openai/gpt-4.1-mini instead (or any gateway shortcut: gr, c, n, ...).
PRIMARY_PROVIDER = os.environ.get("AGENT_PROVIDER", "g")


def role_temperature() -> float:
    """Gemini 3 loops at low temperature, so it needs ~1.0. Other workers
    (e.g. gpt-4.1-mini) behave better at a lower temperature."""
    return 1.0 if PRIMARY_PROVIDER in ("g", "gem", "gemini") else 0.4

# Load the vendored gateway client (llm_gatewayV3/client.py) by file path under
# a unique module name. We deliberately do NOT add llm_gatewayV3/ to sys.path:
# that directory has its own schemas.py / cache.py etc. that would shadow our
# top-level modules (notably schemas.py) depending on import order.
_GW_DIR = Path(__file__).parent / "llm_gatewayV3"
_spec = importlib.util.spec_from_file_location("_gw_client", _GW_DIR / "client.py")
_client_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_client_mod)
LLM = _client_mod.LLM


class GatewayError(RuntimeError):
    pass


_llm = LLM()  # defaults to LLM_GATEWAY_V3_URL or http://localhost:8101


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # drop the opening fence line (``` or ```json) and the trailing fence
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1 :]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[: -3]
    return t.strip()


def _coerce_json(text: str) -> dict:
    """Best-effort JSON decode of an LLM text payload. We only ever apply this
    when the gateway did not return a `parsed` object. This is deserialization
    of a declared-JSON response, not regex parsing of free-form output."""
    candidate = _strip_code_fence(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # The response_format'd output should be pure JSON; if a model wrapped
        # it in prose, take the outermost {...} span and decode that.
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(candidate[start : end + 1])
        raise


# Transient HTTP statuses we retry (free-tier rate limits / pool churn).
_RETRY_STATUS = {429, 502, 503}
_BACKOFF_SECONDS = (3.0, 6.0, 9.0)


def _run_with_retry(call_fn, provider: Optional[str], fallback_provider: Optional[str]) -> dict:
    """Run `call_fn(prov)` with backoff on transient errors. If a ROUTED call
    (provider is None) exhausts retries and `fallback_provider` is set, make one
    final pinned attempt. Raises GatewayError on total failure."""
    raw: Optional[dict] = None
    last_err: Optional[Exception] = None
    for attempt in range(len(_BACKOFF_SECONDS) + 1):
        try:
            raw = call_fn(provider)
            break
        except httpx.HTTPStatusError as e:
            last_err = e
            if e.response.status_code in _RETRY_STATUS and attempt < len(_BACKOFF_SECONDS):
                time.sleep(_BACKOFF_SECONDS[attempt])
                continue
            break
        except Exception as e:  # connection refused, timeout, etc.
            last_err = e
            break

    if raw is None:
        if fallback_provider and provider is None:
            try:
                raw = call_fn(fallback_provider)
            except Exception as e:
                last_err = e
        if raw is None:
            raise GatewayError(
                f"LLM Gateway V3 call failed ({type(last_err).__name__}: {last_err}). "
                f"Verify the gateway is up on {_llm.base_url} "
                f"(`curl -s {_llm.base_url}/v1/status`). On a single-worker pool, "
                f"free-tier rate limits can cause transient 503s — add a second "
                f"worker key (e.g. GROQ_API_KEY / GITHUB_ACCESS_TOKEN) for redundancy."
            ) from last_err
    return raw


def structured(
    *,
    system: str,
    prompt: str,
    schema_model: Type[BaseModel],
    provider: Optional[str] = None,
    auto_route: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    strict: bool = False,
    fallback_provider: Optional[str] = None,
) -> tuple[dict, dict]:
    """One structured (response_format) LLM call. Returns (parsed_dict, raw).
    Used by Perception and Memory, whose outputs are pure data objects."""
    response_format: dict[str, Any] = {
        "type": "json_schema",
        "name": schema_model.__name__,
        "schema": schema_model.model_json_schema(),
        "strict": strict,
    }

    def _call(prov: Optional[str]) -> dict:
        return _llm.chat(
            prompt, system=system, provider=prov, auto_route=auto_route,
            temperature=temperature, max_tokens=max_tokens,
            response_format=response_format,
        )

    raw = _run_with_retry(_call, provider, fallback_provider)
    parsed = raw.get("parsed")
    if not isinstance(parsed, dict):
        text = raw.get("text", "") or ""
        try:
            parsed = _coerce_json(text)
        except json.JSONDecodeError as e:
            raise GatewayError(
                "LLM did not return parseable structured output for "
                f"{schema_model.__name__}. Raw text was:\n{text[:500]}"
            ) from e
    return parsed, raw


def tool_or_text(
    *,
    system: str,
    prompt: str,
    tools: list[dict],
    provider: Optional[str] = None,
    auto_route: Optional[str] = None,
    temperature: float = 0.4,
    max_tokens: int = 2048,
    fallback_provider: Optional[str] = None,
) -> tuple[Optional[dict], str, dict]:
    """One NATIVE tool-calling LLM call (used by Decision). The model either
    returns a tool call or plain text. Returns (tool_call_or_None, text, raw).

    Native function-calling is the right mechanism for tool selection: providers
    fill arguments as a real object, so we avoid the empty-`{}` / mangled-name
    failures that response_format JSON produces for an open arguments dict. The
    boundary the loop sees is still a Pydantic ToolCall (built in decision.py).
    """
    def _call(prov: Optional[str]) -> dict:
        return _llm.chat(
            prompt, system=system, provider=prov, auto_route=auto_route,
            temperature=temperature, max_tokens=max_tokens,
            tools=tools, tool_choice="auto",
        )

    raw = _run_with_retry(_call, provider, fallback_provider)
    tool_calls = raw.get("tool_calls") or []
    first = tool_calls[0] if tool_calls else None
    text = raw.get("text", "") or ""
    return first, text, raw
