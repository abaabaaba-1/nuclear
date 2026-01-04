import hashlib
import json
from typing import Any, Optional


def canonicalize_json_obj(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))
    except Exception:
        return None


def parse_json_dict(s: str) -> Optional[dict]:
    if not isinstance(s, str):
        return None
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def canonicalize_json_str(s: str) -> Optional[str]:
    obj = parse_json_dict(s)
    return canonicalize_json_obj(obj)


def md5_hex(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    return hashlib.md5(s.encode("utf-8")).hexdigest()
