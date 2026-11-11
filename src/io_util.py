import json, os, hashlib, time
from pathlib import Path

def save_json(path, data, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent) + "\n")
    tmp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text())

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))

def sha256_json(data) -> str:
    return sha256_text(json.dumps(data, sort_keys=True, separators=(",", ":")))

def x_sha256(x) -> str:
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def append_jsonl(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")

def now_shanghai():
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")

def pin_source_files(src_dir):
    rows = []
    for p in sorted(Path(src_dir).rglob("*")):
        if p.is_file() and p.suffix in {".py", ".md", ".json"} and p.name != "__pycache__":
            rows.append(dict(path=str(p.relative_to(src_dir.parent)), sha256=sha256_file(p), bytes=p.stat().st_size))
    return rows
