from __future__ import annotations
import json
import csv
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Optional

def _to_jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return str(x)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(row)) + "\n")

def append_csv(path: str, row: Dict[str, Any], header: Optional[Iterable[str]] = None) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    file_exists = os.path.exists(path)
    if header is None:
        header = list(row.keys())
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(header))
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k) for k in header})