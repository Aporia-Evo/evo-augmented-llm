from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any


def stable_json_dumps(payload: Any) -> str:
    if is_dataclass(payload):
        payload = asdict(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

