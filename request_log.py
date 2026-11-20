"""Request logs for the proxy: metadata by default, message content only when enabled.

Requests carry private text (RustyMail sends email content), so by default a log records
only when, which model, which endpoint and how big the request was. Full request and
streamed-response bodies are written only with REQUEST_LOG_BODIES=true, for debugging.
Files older than the retention period are deleted; the folder and files are private.
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

PRUNE_INTERVAL_SECONDS = 3600
SECONDS_PER_DAY = 86400


def env_flag(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def request_metadata(endpoint: str, body: dict, source: str) -> dict:
    """What a metadata-only log keeps: no message text, tool definitions or options."""
    return {
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "source": source,
        "model": body.get("model", "unknown"),
        "stream": bool(body.get("stream", False)),
        "message_count": len(body.get("messages", []) or []),
        "tool_count": len(body.get("tools", []) or []),
    }


class RequestLogger:
    """Writes request logs into one private folder and prunes old ones."""

    def __init__(self, log_dir: Path, log_bodies: bool, retention_days: int):
        self.log_dir = Path(log_dir)
        self.log_bodies = log_bodies
        self.retention_seconds = retention_days * SECONDS_PER_DAY
        self.last_prune = 0.0
        self.log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.log_dir.chmod(0o700)

    @classmethod
    def from_env(cls, default_dir: Path) -> "RequestLogger":
        return cls(
            log_dir=Path(os.environ.get("REQUEST_LOG_DIR") or default_dir),
            log_bodies=env_flag(os.environ.get("REQUEST_LOG_BODIES")),
            retention_days=int(os.environ.get("REQUEST_LOG_RETENTION_DAYS", "7")),
        )

    def log_request(self, endpoint: str, body: dict, source: str = "unknown") -> Path:
        self.prune_if_due()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        model = body.get("model", "unknown")
        path = self.log_dir / f"{timestamp}_{model.replace('/', '_')}_{endpoint.replace('/', '_')}.json"
        record = request_metadata(endpoint, body, source)
        if self.log_bodies:
            record["body"] = body
        self.write_private(path, json.dumps(record, indent=2))
        return path

    def response_log_path(self, model: str) -> Optional[Path]:
        """Where to save a streamed response, or None when bodies aren't logged."""
        if not self.log_bodies:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.log_dir / f"{timestamp}_{model.replace('/', '_')}_response.jsonl"
        self.write_private(path, "")
        return path

    def prune_if_due(self):
        if time.time() - self.last_prune >= PRUNE_INTERVAL_SECONDS:
            self.prune()

    def prune(self, now: Optional[float] = None) -> int:
        """Delete log files older than the retention period; return how many."""
        now = time.time() if now is None else now
        self.last_prune = now
        removed = 0
        for path in self.log_dir.iterdir():
            if path.is_file() and now - path.stat().st_mtime > self.retention_seconds:
                path.unlink()
                removed += 1
        return removed

    @staticmethod
    def write_private(path: Path, text: str):
        """Create or truncate a file readable only by its owner."""
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(text)
