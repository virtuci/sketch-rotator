"""Session files on the volume shared by the gateway and the GPU workers."""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

_ID = re.compile(r"^[0-9a-f]{32}$")


class SessionNotFound(KeyError):
    pass


class Storage:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir) / "sessions"
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self) -> str:
        sid = uuid.uuid4().hex
        (self.root / sid).mkdir()
        return sid

    def dir(self, session_id: str) -> Path:
        # The id reaches us from a URL; only ever join a strict hex id onto the root.
        if not _ID.match(session_id):
            raise SessionNotFound(session_id)
        path = self.root / session_id
        if not path.is_dir():
            raise SessionNotFound(session_id)
        return path

    def url(self, session_id: str, *parts: str) -> str:
        return "/files/sessions/" + "/".join((session_id, *parts))

    def write_meta(self, session_id: str, meta: dict) -> None:
        (self.dir(session_id) / "session.json").write_text(json.dumps(meta, indent=2))

    def read_meta(self, session_id: str) -> dict:
        path = self.dir(session_id) / "session.json"
        if not path.exists():
            raise SessionNotFound(session_id)
        return json.loads(path.read_text())
