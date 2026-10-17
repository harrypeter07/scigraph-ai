"""Export Entire Codebase and Copy to Windows Clipboard.

Collects all code, database scripts, API routes, ML pipelines, tests, web dashboard,
reports, and README, and copies the entire concatenated bundle directly to the clipboard.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".gemini",
    "legacy_archive",
    "node_modules",
    "data",  # skip parquet binaries
    ".venv",
    "venv",
}

INCLUDE_EXTENSIONS = {
    ".py",
    ".sh",
    ".html",
    ".css",
    ".js",
    ".md",
    ".json",
    ".toml",
    ".txt",
    ".ini",
    ".yml",
    ".yaml",
}

IMPORTANT_FILES = [
    "README.md",
    "docs/THE_TRUTH_ABOUT_ACCURACY.md",
    "requirements.txt",
    "pytest.ini",
    "Dockerfile",
    "railway.toml",
    "reports/benchmark_latest.json",
    "reports/heldout_predictions_summary.json",
    "reports/dense_corpus_stats.json",
    "reports/accuracy_improvement/ACCURACY_IMPROVEMENT_REPORT.md",
    "reports/accuracy_improvement/THE_TRUTH_ABOUT_ACCURACY.md",
    "reports/accuracy_improvement/phase0_audit.json",
    "reports/accuracy_improvement/ceiling_nb.json",
    "reports/accuracy_improvement/ceiling.json",
    "reports/accuracy_improvement/feature_ladder.json",
    "reports/accuracy_improvement/embeddings.json",
    "reports/accuracy_improvement/model_ladder.json",
    "reports/accuracy_improvement/learning_curve.json",
    "reports/accuracy_improvement/alternative_tasks.json",
]


def collect_files() -> List[Path]:
    files = []

    # 1. Root & config files
    for imp in IMPORTANT_FILES:
        p = REPO_ROOT / imp
        if p.exists() and p.is_file():
            files.append(p)

    # 2. Walk code directories
    for target_dir in ["api", "docs", "ml", "scripts", "tests", "web"]:
        dir_path = REPO_ROOT / target_dir
        if not dir_path.exists():
            continue
        for root, dirs, filenames in os.walk(dir_path):
            # Prune excluded dirs
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in sorted(filenames):
                p = Path(root) / f
                if p.suffix in INCLUDE_EXTENSIONS:
                    # Skip large parquet / cache files
                    if "cache" in p.parts:
                        continue
                    if p not in files:
                        files.append(p)

    return sorted(files, key=lambda x: str(x.relative_to(REPO_ROOT)))


def bundle_codebase(files: List[Path]) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append("SCIGRAPH AI 2.0 — COMPLETE CODEBASE & DOCUMENTATION MANIFEST")
    lines.append("Includes: API, ML Pipelines, Verification Ledger, GNN Models, Tests, Web Dashboard, README")
    lines.append(f"Total Files Bundled: {len(files)}")
    lines.append("=" * 100)
    lines.append("\nTABLE OF CONTENTS:")
    for i, f in enumerate(files, 1):
        rel = f.relative_to(REPO_ROOT).as_posix()
        lines.append(f"  [{i:02d}] {rel}")
    lines.append("\n" + "=" * 100 + "\n")

    for f in files:
        rel = f.relative_to(REPO_ROOT).as_posix()
        lines.append("\n" + "#" * 100)
        lines.append(f"# START FILE: {rel}")
        lines.append("#" * 100 + "\n")
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fp:
                content = fp.read()
            lines.append(content)
        except Exception as e:
            lines.append(f"[ERROR READING FILE {rel}: {e}]")
        lines.append("\n" + "#" * 100)
        lines.append(f"# END FILE: {rel}")
        lines.append("#" * 100 + "\n")

    return "\n".join(lines)


def copy_to_clipboard(bundle_text: str):
    # Save bundle to file first
    bundle_path = REPO_ROOT / "reports" / "full_codebase_bundle.txt"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bundle_path, "w", encoding="utf-8") as f:
        f.write(bundle_text)

    print(f"Saved complete codebase bundle to: {bundle_path} ({len(bundle_text):,} characters)")

    # Use PowerShell Set-Clipboard
    ps_cmd = f"Get-Content -Raw -Encoding UTF8 -Path '{bundle_path}' | Set-Clipboard"
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=True)
        print("[SUCCESS] Successfully copied entire codebase to your Windows clipboard!")
    except Exception as e:
        print(f"[WARN] PowerShell Set-Clipboard returned: {e}. Bundle file is preserved at {bundle_path}")


def main():
    files = collect_files()
    print(f"Found {len(files)} source & documentation files to bundle:")
    for f in files:
        print(f"  - {f.relative_to(REPO_ROOT).as_posix()}")

    bundle_str = bundle_codebase(files)
    copy_to_clipboard(bundle_str)


if __name__ == "__main__":
    main()
