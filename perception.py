"""
perception.py — the orchestrator/CEO and verifier. Runs every iteration.

Reads the query + memory hits + history + prior goals and emits a fresh goal
list with `done` flags and optional artifact attachments. On the first run it
DECOMPOSES the task; on later runs it only flips `done` flags (verifying a goal
the moment a satisfying action/answer appears in history) and sets attachments.

One LLM call, forced to Gemini (provider="g"), temperature=1.0.

Anti-hallucination mapping happens HERE, in Python, not in the model:
  * Positional goal identity — the model emits goals with NO id; ids are carried
    by POSITION from the prior goal list.
  * Integer artifact indices — artifact-bearing memory hits are shown with an
    integer index; the model emits `artifact_index`, which we map back to a real
    handle, guarded by artifacts.exists() (a hallucinated index is dropped).
  * Force-attach safety net — for a synthesis-type goal with an artifact present
    in the hits, auto-attach the most recent artifact even if the model didn't.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from artifacts import ArtifactStore
from gateway import PRIMARY_PROVIDER, role_temperature, structured
from schemas import Goal, MemoryItem, Observation, PerceptionOutput

_PROMPT = (Path(__file__).parent / "prompts" / "perception.txt").read_text(encoding="utf-8")

_SYNTH_WORDS = (
    "synthesise", "synthesize", "extract", "list", "compare", "decide",
    "summarise", "summarize",
)


def _is_synthesis(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in _SYNTH_WORDS)


class Perception:
    def __init__(self, artifacts: ArtifactStore):
        self.artifacts = artifacts

    def observe(
        self,
        query: str,
        hits: list[MemoryItem],
        history: list[dict],
        prior_goals: list[Goal],
        run_id: str,
    ) -> Observation:
        # Build the integer-index map over artifact-bearing hits.
        artifact_hits = [m for m in hits if m.artifact_id and self.artifacts.exists(m.artifact_id)]
        index_to_handle: dict[int, str] = {i: m.artifact_id for i, m in enumerate(artifact_hits)}  # type: ignore[misc]
        most_recent_handle: Optional[str] = None
        if artifact_hits:
            most_recent_handle = max(artifact_hits, key=lambda m: m.created_at).artifact_id

        prompt = self._build_prompt(query, hits, artifact_hits, history, prior_goals)

        parsed, _ = structured(
            system=_PROMPT,
            prompt=prompt,
            schema_model=PerceptionOutput,
            provider=PRIMARY_PROVIDER,   # pinned worker (default Gemini; AGENT_PROVIDER)
            auto_route="perception",
            temperature=role_temperature(),
            strict=False,
        )
        po = PerceptionOutput.model_validate(parsed)

        goals: list[Goal] = []
        for i, pg in enumerate(po.goals):
            gid = prior_goals[i].id if i < len(prior_goals) else f"g{i + 1}"
            attach: Optional[str] = None
            # 1) explicit integer index from the model -> handle (guarded)
            if pg.artifact_index is not None and pg.artifact_index in index_to_handle:
                cand = index_to_handle[pg.artifact_index]
                if self.artifacts.exists(cand):
                    attach = cand
            # 2) force-attach safety net for synthesis goals
            if attach is None and not pg.done and _is_synthesis(pg.text) and most_recent_handle:
                if self.artifacts.exists(most_recent_handle):
                    attach = most_recent_handle
            goals.append(Goal(id=gid, text=pg.text, done=pg.done, attach_artifact_id=attach))

        return Observation(goals=goals)

    # -- prompt assembly ------------------------------------------------------
    def _build_prompt(
        self,
        query: str,
        hits: list[MemoryItem],
        artifact_hits: list[MemoryItem],
        history: list[dict],
        prior_goals: list[Goal],
    ) -> str:
        if prior_goals:
            pg_lines = "\n".join(
                f"  {i}. [{'x' if g.done else ' '}] {g.text}" for i, g in enumerate(prior_goals)
            )
            prior_block = (
                "PRIOR GOALS (keep the SAME goals in the SAME order; only flip "
                "done flags — do NOT add, remove, or reorder goals):\n" + pg_lines
            )
        else:
            prior_block = (
                "PRIOR GOALS: (none — this is the first run. DECOMPOSE the query "
                "into a short ordered list of concrete sub-goals.)"
            )

        # Artifact-bearing hits get integer indices the model may reference.
        if artifact_hits:
            ah_lines = "\n".join(
                f"  [artifact_index={i}] {m.descriptor}" for i, m in enumerate(artifact_hits)
            )
            artifact_block = "INDEXED ARTIFACTS IN MEMORY:\n" + ah_lines
        else:
            artifact_block = "INDEXED ARTIFACTS IN MEMORY: (none)"

        hit_lines = "\n".join(f"  - [{m.kind}] {m.descriptor}" for m in hits) or "  (none)"

        hist_lines = []
        for ev in history[-10:]:
            if ev.get("type") == "action":
                hist_lines.append(f"  - action: {ev.get('descriptor', '')[:240]}")
            elif ev.get("type") == "answer":
                hist_lines.append(f"  - answer: {ev.get('text', '')[:240]}")
        hist_block = "\n".join(hist_lines) or "  (none)"

        return (
            f"USER QUERY:\n{query}\n\n"
            f"{prior_block}\n\n"
            f"MEMORY HITS:\n{hit_lines}\n\n"
            f"{artifact_block}\n\n"
            f"HISTORY THIS RUN:\n{hist_block}\n"
        )
