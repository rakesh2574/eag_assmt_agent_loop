"""
schemas.py — every cross-role boundary in the system is one of these Pydantic v2
models. There is NO free-form dict passing between roles.

Each class is simultaneously:
  * the contract (what one role hands the next),
  * the validator (Pydantic enforces it),
  * the JSON Schema sent to the LLM via `response_format`,
  * and the on-disk persistence format (model_dump_json / model_validate).

Anti-hallucination split
------------------------
The model that Perception EMITS (`PerceptionOutput`/`PerceivedGoal`) deliberately
has NO goal-id string field and NO raw `art:` handle field. Goals are identified
by POSITION; artifacts by integer INDEX. The loop maps positions->ids and
indices->handles. A weak router-tier model therefore has nowhere to invent a
stale id or hallucinate a handle. The INTERNAL models (`Goal`, `Observation`)
do carry the real ids/handles, assigned by the orchestration loop.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
class MemoryItem(BaseModel):
    """One unit of memory. Read with pure keyword search; written either by a
    single classify call (free-form) or structurally (tool outcomes)."""

    id: str
    kind: Literal["fact", "preference", "tool_outcome", "scratchpad"]
    keywords: list[str] = Field(default_factory=list)
    descriptor: str = ""
    value: dict = Field(default_factory=dict)
    artifact_id: Optional[str] = None
    source: str = ""
    run_id: str = ""
    goal_id: Optional[str] = None
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=_utcnow)

    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
class Artifact(BaseModel):
    """Metadata sidecar for a stored blob. The raw bytes live in `<id>.bin`;
    this is the `<id>.json`. Perception NEVER sees the bytes — only this meta
    (and only via an integer index, never the raw handle)."""

    id: str  # "art:<sha256-prefix>"
    content_type: str = "text/plain"
    size_bytes: int = 0
    source: str = ""
    descriptor: str = ""


# ---------------------------------------------------------------------------
# Goals  (INTERNAL — carries real ids/handles, owned by the loop/Perception)
# ---------------------------------------------------------------------------
class Goal(BaseModel):
    """A single sub-goal. `id` is assigned by the loop from POSITION, never by
    the model. `attach_artifact_id` is a resolved handle (or None)."""

    id: str
    text: str  # short imperative
    done: bool = False
    attach_artifact_id: Optional[str] = None


class Observation(BaseModel):
    """Perception's verified view of the goal state, returned each iteration."""

    goals: list[Goal] = Field(default_factory=list)

    def all_done(self) -> bool:
        return len(self.goals) > 0 and all(g.done for g in self.goals)

    def next_unfinished(self) -> Optional[Goal]:
        for g in self.goals:
            if not g.done:
                return g
        return None


# ---------------------------------------------------------------------------
# Perception WIRE schema  (what the LLM emits via response_format)
# ---------------------------------------------------------------------------
class PerceivedGoal(BaseModel):
    """A goal as the Perception model is allowed to express it.

    NOTE the absence of any id field: goals are identified positionally.
    `artifact_index` is an INTEGER index into the indexed memory-hit artifacts
    presented in the prompt — never a raw `art:...` string. The loop maps the
    integer back to a real handle (and drops it if it doesn't exist)."""

    text: str
    done: bool = False
    artifact_index: Optional[int] = None


class PerceptionOutput(BaseModel):
    """Top-level structured output of a Perception call."""

    goals: list[PerceivedGoal] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------
class ToolCall(BaseModel):
    """Exactly one MCP tool invocation (INTERNAL boundary — arguments is a real
    dict the loop hands to Action)."""

    name: str
    arguments: dict = Field(default_factory=dict)


class DecisionWire(BaseModel):
    """The schema Decision EMITS via response_format.

    `tool_arguments_json` is a JSON OBJECT encoded as a STRING. This is
    deliberate: constrained decoding cannot reliably fill an open `dict` (the
    model returns `{}` because the schema declares no properties), but it fills
    a plain string field freely. decision.py json-decodes it into the real
    ToolCall.arguments dict (JSON deserialization, not regex parsing)."""

    answer: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments_json: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class DecisionOutput(BaseModel):
    """Decision returns EITHER a final answer OR exactly one tool call —
    never both, never two tools, never narration.

    The model may, being a weak router-tier model, occasionally populate both
    fields. The `mode="after"` validator normalizes to the invariant rather
    than crashing the loop: a populated tool name wins; otherwise the answer
    stands; an empty output raises (caller treats it as a failed step)."""

    answer: Optional[str] = None
    tool_call: Optional[ToolCall] = None

    model_config = ConfigDict(extra="ignore")

    @property
    def is_answer(self) -> bool:
        return self.tool_call is None and self.answer is not None

    def model_post_init(self, _ctx) -> None:  # normalize the invariant
        has_tool = self.tool_call is not None and bool(
            (self.tool_call.name or "").strip()
        )
        if has_tool:
            # A real tool call wins; an answer alongside it is dropped.
            object.__setattr__(self, "answer", None)
        else:
            # No usable tool call: it must be an answer.
            object.__setattr__(self, "tool_call", None)
            if self.answer is None:
                object.__setattr__(self, "answer", "")
