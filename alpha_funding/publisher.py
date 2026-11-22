"""Publish through the ONE existing door: POST /webhook/strategy.

The row is deliberately inert:
  - mode = "shadow"
  - rsi_lower = 0.0, rsi_upper = 100.0  -> CanArm() rejects (0,100 exclusive)

The Go executor therefore stores the row and shows it in the research panel,
but can never arm it. No Go code changes, no new endpoint, no new schema.
"""
import json
import os
from datetime import datetime, timezone

import requests

from . import config
from .signal import executor_symbol, build_research


def payload_for(symbol, result, retired=False):
    """Build the exact payload shape the executor already understands."""
    return {
        "symbol": executor_symbol(symbol),
        "direction": "SHORT" if result["fire"] else "LONG",
        "leverage": f"{int(config.LEVERAGE)}x",
        "rsi_lower": 0.0,
        "rsi_upper": 100.0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "retired": bool(retired),
        "mode": "shadow",
        "research": build_research(result, symbol),
    }


def post(payload, dry_run=False, url=None, timeout=None):
    url = url or config.WEBHOOK_URL
    timeout = timeout or config.WEBHOOK_TIMEOUT
    if dry_run:
        return {"ok": True, "dry_run": True, "payload": payload}
    try:
        r = requests.post(url, json=payload,
                          headers={"Content-Type": "application/json"}, timeout=timeout)
        return {"ok": r.status_code < 300, "status": r.status_code,
                "body": r.text[:300]}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def load_state(path=None):
    path = path or config.STATE_PATH
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_state(state, path=None):
    path = path or config.STATE_PATH
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def append_audit(rows, path=None):
    path = path or config.AUDIT_PATH
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
