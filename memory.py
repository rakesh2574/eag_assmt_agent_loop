"""
memory.py — a typed memory service that lives BESIDE the loop (it is not a
cognitive role with a turn; the loop calls it).

Reads are pure Python keyword search — NO LLM. Only `remember()` (free-form
writes) spends one classify call, pinned to Gemini. `record_outcome()` is
structural and uses no LLM. Non-scratchpad items persist to state/memory.json
so facts survive across runs.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from gateway import PRIMARY_PROVIDER, role_temperature, structured
from schemas import MemoryItem, ToolCall

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "me", "my", "i", "you", "your", "it", "that", "this",
    "with", "at", "by", "from", "as", "give", "tell", "find", "get", "what",
    "when", "where", "who", "how", "please", "about", "into", "do", "does",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS]


def _short_id() -> str:
    return uuid.uuid4().hex[:12]


class _Classification(BaseModel):
    """Schema for the single classify call in remember()."""
    kind: str = "scratchpad"  # fact | preference | tool_outcome | scratchpad
    keywords: list[str] = Field(default_factory=list)
    descriptor: str = ""
    value: dict = Field(default_factory=dict)


_CLASSIFY_SYSTEM = (
    "You classify a single piece of text into a memory record. Output JSON only.\n"
    "Fields:\n"
    "  kind: one of 'fact' (durable truth, e.g. a date/name the user states), "
    "'preference' (a changeable taste), 'tool_outcome' (a record of a tool run), "
    "or 'scratchpad' (a transient note, e.g. a question or instruction).\n"
    "  keywords: 3-8 lowercase salient search tokens (names, dates, nouns).\n"
    "  descriptor: one short sentence summarizing the content.\n"
    "  value: a small structured object capturing the key data (e.g. "
    "{\"subject\":\"mom birthday\",\"date\":\"2026-05-15\"}).\n"
    "If the text states a concrete personal fact the user wants remembered "
    "(a birthday, a name, a date), use kind='fact' and put the data in value."
)

_VALID_KINDS = {"fact", "preference", "tool_outcome", "scratchpad"}


class Memory:
    def __init__(self, state_dir: Path):
        self.path = Path(state_dir) / "memory.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: list[MemoryItem] = []
        self._load()

    # -- persistence ----------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            import json

            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for d in raw:
            try:
                self._items.append(MemoryItem.model_validate(d))
            except Exception:
                continue

    def _persist(self) -> None:
        # scratchpad is run-scoped: never written to disk.
        durable = [m for m in self._items if m.kind != "scratchpad"]
        import json

        self.path.write_text(
            json.dumps([m.model_dump(mode="json") for m in durable], indent=2),
            encoding="utf-8",
        )

    # -- reads (NO LLM) -------------------------------------------------------
    def read(
        self,
        query: str,
        history: Optional[list[dict]] = None,
        kinds: Optional[list[str]] = None,
        top_k: int = 8,
    ) -> list[MemoryItem]:
        """Lowercase-token intersection over keywords + descriptor tokens.
        Search tokens come from the query plus recent history text, so a fact
        written by Action earlier in the run can be read back here."""
        search_tokens = set(_tokens(query))
        for ev in (history or [])[-4:]:
            search_tokens |= set(_tokens(str(ev.get("descriptor", ""))))
            search_tokens |= set(_tokens(str(ev.get("text", ""))))

        scored: list[tuple[int, float, MemoryItem]] = []
        for m in self._items:
            if kinds and m.kind not in kinds:
                continue
            item_tokens = set(t.lower() for t in m.keywords) | set(_tokens(m.descriptor))
            overlap = len(search_tokens & item_tokens)
            if overlap == 0:
                continue
            scored.append((overlap, m.confidence, m))
        scored.sort(key=lambda x: (x[0], x[1], x[2].created_at), reverse=True)
        return [m for _, _, m in scored[:top_k]]

    def filter(
        self,
        kinds: Optional[list[str]] = None,
        goal_id: Optional[str] = None,
        recent: Optional[int] = None,
    ) -> list[MemoryItem]:
        """Structured filter — NO LLM, no keyword search."""
        items = self._items
        if kinds:
            items = [m for m in items if m.kind in kinds]
        if goal_id is not None:
            items = [m for m in items if m.goal_id == goal_id]
        items = sorted(items, key=lambda m: m.created_at, reverse=True)
        if recent is not None:
            items = items[:recent]
        return items

    # -- writes ---------------------------------------------------------------
    def remember(
        self, raw_text: str, *, source: str, run_id: str, goal_id: Optional[str] = None
    ) -> MemoryItem:
        """ONE classify call (Gemini) to structure a free-form write."""
        parsed, _ = structured(
            system=_CLASSIFY_SYSTEM,
            prompt=raw_text,
            schema_model=_Classification,
            provider=PRIMARY_PROVIDER,   # pinned worker (wins over auto_route)
            auto_route="memory",         # cognitive-layer hint, for the dashboard
            temperature=role_temperature(),
            strict=False,
        )
        c = _Classification.model_validate(parsed)
        kind = c.kind if c.kind in _VALID_KINDS else "scratchpad"
        # A user query is never a tool run. Keep genuine facts/preferences a user
        # states (e.g. "mom's birthday is 15 May 2026"), but never let an
        # instruction be stored as a 'tool_outcome' — that would masquerade as a
        # completed action in Perception's memory hits and skip the real step.
        if source == "user_query" and kind == "tool_outcome":
            kind = "scratchpad"
        keywords = [k.lower() for k in c.keywords] or _tokens(raw_text)[:8]
        item = MemoryItem(
            id=_short_id(),
            kind=kind,  # type: ignore[arg-type]
            keywords=keywords,
            descriptor=c.descriptor or raw_text[:120],
            value=c.value or {"text": raw_text},
            artifact_id=None,
            source=source,
            run_id=run_id,
            goal_id=goal_id,
            confidence=0.9,
        )
        self._items.append(item)
        self._persist()
        return item

    def record_outcome(
        self,
        tool_call: ToolCall,
        result_text: str,
        artifact_id: Optional[str],
        run_id: str,
        goal_id: Optional[str] = None,
    ) -> MemoryItem:
        """Structural write — NO LLM. Keywords come from the tool name and its
        argument tokens, so future keyword reads can find this outcome."""
        kw = _tokens(tool_call.name)
        for v in tool_call.arguments.values():
            kw += _tokens(str(v))
        # also fold a few salient tokens from the result so e.g. a weather
        # outcome is findable by "weather"/"forecast".
        kw += _tokens(result_text)[:12]
        item = MemoryItem(
            id=_short_id(),
            kind="tool_outcome",
            keywords=list(dict.fromkeys(kw))[:20],
            descriptor=f"{tool_call.name}({tool_call.arguments}) -> {result_text[:120]}",
            value={
                "tool": tool_call.name,
                "arguments": tool_call.arguments,
                "result_preview": result_text[:400],
                "artifact_id": artifact_id,
            },
            artifact_id=artifact_id,
            source="action",
            run_id=run_id,
            goal_id=goal_id,
            confidence=1.0,
        )
        self._items.append(item)
        self._persist()
        return item
