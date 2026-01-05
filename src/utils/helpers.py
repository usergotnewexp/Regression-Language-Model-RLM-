"""Utility helpers for training and inference."""
import json
from pathlib import Path


def save_json(obj, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        json.dump(obj, f)


def load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, 'r') as f:
        return json.load(f)
