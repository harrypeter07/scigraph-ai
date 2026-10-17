"""Generate data/MANIFEST.sha256 containing SHA256 checksums for all Parquet files."""

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    parquet_files = sorted(list(DATA_DIR.rglob("*.parquet")))
    lines = []
    for p in parquet_files:
        rel_path = p.relative_to(REPO_ROOT).as_posix()
        digest = compute_sha256(p)
        lines.append(f"{digest}  {rel_path}")
        print(f"{digest[:12]}...  {rel_path}")

    manifest_path = DATA_DIR / "MANIFEST.sha256"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {len(lines)} file hashes to {manifest_path}")

if __name__ == "__main__":
    main()
