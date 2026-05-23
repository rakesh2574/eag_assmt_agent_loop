"""
artifacts.py — content-addressable blob store.

Any tool result larger than the 4 KB threshold (enforced in action.py) is put
here; Action then returns only a short descriptor. Handles look like
`art:<sha256-prefix>`. Identical content dedupes (same bytes -> same handle).

Two files per artifact under state/artifacts/:
  <prefix>.bin   raw bytes
  <prefix>.json  Artifact metadata

No eviction policy — state/ is wiped between attempts.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from schemas import Artifact

_PREFIX_LEN = 16  # chars of the sha256 hexdigest used in the handle


class ArtifactStore:
    def __init__(self, root: Path):
        self.dir = Path(root) / "artifacts"
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- internal path helpers ------------------------------------------------
    def _prefix(self, handle: str) -> str:
        return handle.split("art:", 1)[-1]

    def _bin_path(self, handle: str) -> Path:
        return self.dir / f"{self._prefix(handle)}.bin"

    def _meta_path(self, handle: str) -> Path:
        return self.dir / f"{self._prefix(handle)}.json"

    # -- public API -----------------------------------------------------------
    def put(
        self,
        blob: bytes,
        *,
        content_type: str = "text/plain",
        source: str = "",
        descriptor: str = "",
    ) -> str:
        digest = hashlib.sha256(blob).hexdigest()[:_PREFIX_LEN]
        handle = f"art:{digest}"
        bin_path = self._bin_path(handle)
        if bin_path.exists():
            return handle  # content-addressable dedupe
        bin_path.write_bytes(blob)
        meta = Artifact(
            id=handle,
            content_type=content_type,
            size_bytes=len(blob),
            source=source,
            descriptor=descriptor,
        )
        self._meta_path(handle).write_text(
            meta.model_dump_json(indent=2), encoding="utf-8"
        )
        return handle

    def exists(self, handle: str) -> bool:
        if not handle or not handle.startswith("art:"):
            return False
        return self._bin_path(handle).exists()

    def get_bytes(self, handle: str) -> bytes:
        return self._bin_path(handle).read_bytes()

    def get_meta(self, handle: str) -> Artifact:
        return Artifact.model_validate_json(
            self._meta_path(handle).read_text(encoding="utf-8")
        )
