"""Durable metadata writes and exact cache compatibility for FineDance batches."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

# Only these exact two builder revisions share generation behavior. Other code
# changes retain normal whole-file cache invalidation.
LEGACY_BUILDER_SHA256 = "97789f97b563cbf7cac4951e245e493aea2120c3b8a4bd5e84fd3a91fa259922"
BOOKKEEPING_BUILDER_SHA256 = "0146aa6d32f3eefa9df60ef53d2684141aa3afb883814e1302823eda03c597e6"


def cache_builder_hash(path):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return LEGACY_BUILDER_SHA256 if digest == BOOKKEEPING_BUILDER_SHA256 else digest


def atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path, payload):
    # Serialize before opening a file, so serialization errors cannot harm the
    # last checkpoint. Flush data to disk before replacing its canonical name.
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_checkpoint(path, key, *, recover):
    if not path.exists():
        return {}, True
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict) or not isinstance(value.get(key), list):
            raise ValueError(f"Expected an object containing a {key} list")
        seen = set()
        for entry in value[key]:
            if not isinstance(entry, dict) or not isinstance(entry.get("source_motion_id"), str):
                raise ValueError("Invalid checkpoint entry")
            name = entry["source_motion_id"]
            if name in seen:
                raise ValueError("Duplicate checkpoint entry")
            seen.add(name)
            if key == "motions" and (entry.get("artifact") != name + ".pkl"
                                     or not isinstance(entry.get("clip"), dict)
                                     or entry["clip"].get("split") not in ("train", "val", "test")):
                raise ValueError("Invalid motion checkpoint entry")
        return value, False
    except (ValueError, UnicodeError) as exc:
        if not recover:
            raise ValueError(f"Damaged output checkpoint {path}; rerun with --resume or --repair-only.") from exc
        backup = path.with_name(f"{path.name}.corrupt.{time.time_ns()}")
        path.replace(backup)
        print(f"Preserved damaged {path.name} as {backup.name}.", flush=True)
        return {}, True
