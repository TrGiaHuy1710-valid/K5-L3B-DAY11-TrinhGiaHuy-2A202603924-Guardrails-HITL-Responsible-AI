"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

MAX_LOG_CHARS = 1000  # cắt bớt text dài (prompt 5000 ký tự) để log gọn


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._pending: dict[str, dict] = {}

    @staticmethod
    def _key(user_id: str, request_id: str | None) -> str:
        return request_id or f"user:{user_id}"

    @staticmethod
    def _scrub(text: str) -> str:
        """Audit log cũng là nơi có thể lộ dữ liệu — redact PII/secret trước khi lưu.

        (Attacker có thể dán "password=admin123" vào input; log không được giữ nguyên.)
        """
        from guardrails.output_guardrails import content_filter

        return content_filter(text or "")["redacted"]

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id (fallback: user_id)."""
        key = self._key(user_id, request_id)
        self._open[key] = time.perf_counter()
        self._pending[key] = {
            "request_id": request_id or key,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            "input": self._scrub(text)[:MAX_LOG_CHARS],
            "input_chars": len(text or ""),
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
        reason: str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        key = self._key(user_id, request_id)
        start = self._open.pop(key, None)
        entry = self._pending.pop(key, None) or {
            "request_id": request_id or key,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            "input": None,
            "input_chars": None,
        }
        entry.update(
            {
                "output": self._scrub(text)[:MAX_LOG_CHARS],
                "blocked": bool(blocked),
                "layer": layer,
                "reason": reason,
                "latency_ms": (
                    round((time.perf_counter() - start) * 1000, 1)
                    if start is not None
                    else None
                ),
                "completed_at": utc_now_iso(),
            }
        )
        self.logs.append(entry)
        return entry

    def summary(self) -> dict:
        by_layer: dict[str, int] = {}
        for log in self.logs:
            if log["blocked"]:
                by_layer[log["layer"] or "unknown"] = by_layer.get(log["layer"] or "unknown", 0) + 1
        return {
            "total": len(self.logs),
            "blocked": sum(1 for log in self.logs if log["blocked"]),
            "blocked_by_layer": by_layer,
        }

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
