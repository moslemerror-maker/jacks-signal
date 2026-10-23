from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path


def quarantine(
    raw_dir: Path,
    filename: str,
    content: bytes,
    *,
    source_url: str,
    trade_date: date,
    http_status: int,
    reason: str,
) -> Path:
    quarantine_dir = raw_dir / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    q_path = quarantine_dir / filename
    q_path.write_bytes(content)
    q_meta_path = q_path.with_suffix(q_path.suffix + ".meta.json")
    q_meta_path.write_text(
        json.dumps(
            {
                "trade_date": trade_date.isoformat(),
                "source_url": source_url,
                "quarantined_at": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "http_status": http_status,
                "content_length": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            },
            indent=2,
        )
    )
    return q_path
