"""Verify that source files copied for this extension have not changed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


EXTENSION_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXTENSION_DIR.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    manifest = json.loads((EXTENSION_DIR / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    failed = False
    for source in manifest["sources"]:
        path = REPO_ROOT / source["path"]
        actual = sha256(path) if path.is_file() else "MISSING"
        expected = source["sha256"].upper()
        status = "OK" if actual == expected else "CHANGED"
        print(f"{status:7} {source['path']}  {actual}")
        failed |= actual != expected
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
