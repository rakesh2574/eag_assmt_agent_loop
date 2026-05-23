"""
action.py — pure dispatch. NO LLM. Awaits the MCP tool, flattens the result to
text, and decides text-vs-artifact at the 4 KB boundary. Returns
(descriptor, artifact_id_or_None).
"""
from __future__ import annotations

import json

from artifacts import ArtifactStore
from schemas import ToolCall

ARTIFACT_THRESHOLD = 4096  # bytes


def _flatten(result) -> str:
    """Turn an MCP CallToolResult into plain text."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else str(block))
    if not parts:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return json.dumps(structured, ensure_ascii=False)
    return "\n".join(parts)


class Action:
    def __init__(self, artifacts: ArtifactStore):
        self.artifacts = artifacts

    async def execute(self, session, tool_call: ToolCall) -> tuple[str, str | None]:
        # Artifact handles are not file paths or URLs — refuse them as args.
        for v in tool_call.arguments.values():
            if isinstance(v, str) and v.startswith("art:"):
                return (f"ERROR: argument '{v}' is an artifact handle, not a path/URL.", None)
        try:
            result = await session.call_tool(tool_call.name, tool_call.arguments)
        except Exception as e:
            return (f"ERROR calling {tool_call.name}: {type(e).__name__}: {e}", None)
        text = _flatten(result)
        blob = text.encode("utf-8")
        if len(blob) > ARTIFACT_THRESHOLD:
            aid = self.artifacts.put(
                blob, content_type="text/plain", source=tool_call.name,
                descriptor=text[:160].replace("\n", " "),
            )
            return (f"[artifact {aid}, {len(blob)} bytes] preview: {text[:200]}", aid)
        return (text, None)
