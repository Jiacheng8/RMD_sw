"""On-disk contracts shared by the reference builders (no GPU imports)."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_save(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as f:
        writer(f)
    os.replace(tmp, path)


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another worker owns {path}") from None
        yield


def load_prompts(root):
    with (Path(root) / "prompts.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    meta = read_json(Path(root) / "prepare.json")
    if digest(rows) != meta["prompt_hash"] or not rows:
        raise ValueError("Frozen prompt table failed its checksum")
    return rows, meta


def prompt_dir(root, p):
    return Path(root) / "renders" / f"{p // 1000:04d}" / f"{p:08d}"


def seed_for(image_id, candidate, base):
    # Stable across rank count, input file order and smoke/full runs.
    return int(digest([image_id, candidate, base])[:15], 16)


def selection(root, p, keep, verify=True):
    from PIL import Image
    folder = prompt_dir(root, p)
    try:
        data = read_json(folder / "selection.json")
        if data["prompt_id"] != p or len(data["images"]) != keep:
            return None
        for k, row in enumerate(data["images"]):
            if row["rank"] != k or row["file"] != f"k{k}.png":
                return None
            path = folder / row["file"]
            if verify:
                with Image.open(path) as image:
                    image.verify()
            elif not path.is_file():
                return None
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None
