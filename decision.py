"""
decision.py — the worker role. Works on ONE goal at a time and never sees the
others. Returns EITHER a final answer (plain text) OR exactly one tool call —
never both, never two tools, never narration. One LLM call via
`auto_route="decision"`.

The full bytes of an attached artifact (e.g. a 250 KB page) would blow past the
router's HUGE ceiling and 503. So: we cap the attachment to a safe window and,
when the assembled prompt is still large, use the documented escape hatch
(provider="g", Gemini long context) instead of the router. This keeps every
call on the gateway while staying inside what a free worker can actually serve.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from gateway import PRIMARY_PROVIDER, tool_or_text
from schemas import Artifact, DecisionOutput, Goal, MemoryItem, ToolCall

_PROMPT = (Path(__file__).parent / "prompts" / "decision.txt").read_text(encoding="utf-8")

# Window of attached artifact text fed to Decision. Default suits long-context
# workers (Gemini). For GitHub Models' gpt-4.1-mini (8K input cap) set
# MAX_ATTACH_CHARS=18000 in .env so the prompt stays under the cap.
MAX_ATTACH_CHARS = int(os.environ.get("MAX_ATTACH_CHARS", "48000"))
_ROUTER_BYPASS_WORDS = 7500  # est. tokens above which we pin the worker vs. routing


def _est_tokens(text: str) -> float:
    return len(text.split()) * 1.4


def _format_tools(tools: list[dict]) -> str:
    lines = []
    for t in tools:
        schema = t.get("input_schema", {}) or {}
        props = list((schema.get("properties") or {}).keys())
        req = schema.get("required") or []
        lines.append(
            f"- {t['name']}({', '.join(props)}) "
            f"[required: {', '.join(req) or 'none'}]: {t.get('description', '').strip()}"
        )
    return "\n".join(lines)


def _format_hits(hits: list[MemoryItem]) -> str:
    if not hits:
        return "(none)"
    out = []
    for m in hits:
        line = f"- [{m.kind}] {m.descriptor}"
        if m.value:
            line += f"  value={json.dumps(m.value, ensure_ascii=False)[:300]}"
        out.append(line)
    return "\n".join(out)


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    out = []
    for ev in history[-8:]:
        if ev.get("type") == "action":
            out.append(f"- action: {ev.get('descriptor', '')[:300]}")
        elif ev.get("type") == "answer":
            out.append(f"- answer: {ev.get('text', '')[:300]}")
    return "\n".join(out) or "(none)"


class Decision:
    def next_step(
        self,
        goal: Goal,
        hits: list[MemoryItem],
        attached: list[tuple[Artifact, str]],
        history: list[dict],
        tools: list[dict],
        query: str = "",
    ) -> DecisionOutput:
        attach_block = "(none)"
        if attached:
            chunks = []
            for meta, text in attached:
                body = text[:MAX_ATTACH_CHARS]
                truncated = " [TRUNCATED]" if len(text) > MAX_ATTACH_CHARS else ""
                chunks.append(
                    f"--- artifact {meta.id} ({meta.size_bytes} bytes){truncated} ---\n{body}"
                )
            attach_block = "\n\n".join(chunks)

        prompt = (
            f"OVERALL TASK (for grounding only — act ONLY on the GOAL below, but "
            f"take concrete details like URLs / exact search phrases from here):\n{query}\n\n"
            f"GOAL (work only on this one):\n{goal.text}\n\n"
            f"AVAILABLE TOOLS:\n{_format_tools(tools)}\n\n"
            f"RELEVANT MEMORY:\n{_format_hits(hits)}\n\n"
            f"RECENT HISTORY:\n{_format_history(history)}\n\n"
            f"ATTACHED ARTIFACTS:\n{attach_block}\n"
        )

        # Pick the call path: route normally, but if the prompt is large
        # (a big attachment) bypass the router to avoid a HUGE 503.
        if _est_tokens(prompt) > _ROUTER_BYPASS_WORDS:
            provider: Optional[str] = PRIMARY_PROVIDER
            auto_route: Optional[str] = None
        else:
            provider = None
            auto_route = "decision"

        first, text, _ = tool_or_text(
            system=_PROMPT,
            prompt=prompt,
            tools=tools,
            provider=provider,
            auto_route=auto_route,
            temperature=0.4,
            max_tokens=2048,
            fallback_provider=PRIMARY_PROVIDER,  # routed worker down -> pinned worker
        )
        if first and (first.get("name") or "").strip():
            args = first.get("arguments")
            if not isinstance(args, dict):
                args = {}
            return DecisionOutput(tool_call=ToolCall(name=first["name"].strip(), arguments=args))
        return DecisionOutput(answer=text or "")
