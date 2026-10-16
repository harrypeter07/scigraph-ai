"""Dynamic script to collect and copy EVERY SINGLE active codebase file in SciGraph AI 2.0 to Windows Clipboard:
- Frontend: web/index.html (complete updated 5-tab verification UI)
- Backend & REST API: api/**/*.py (main.py, config.py, openalex_client.py, routers, schemas)
- Machine Learning & Serving: ml/**/*.py (primary model, inference service, GNN, ego_graph, feature builder, attribution, scientometrics)
- Test Suite: tests/**/*.py (test_audit_verification.py and all 39 tests)
- Operational & Audit Scripts: scripts/**/*.py (update_ui, make_numbers, audit_leakage, etc.)
- Scientific Reports & Proofs: reports/*.json, reports/*.md, reports/proof/*.txt, docs/*.md, docs/*.json
- Root Configs: README.md, pytest.ini, .gitignore

Saves to 'ALL_CODEBASE_FILES.txt' and copies directly to the Windows Clipboard via PowerShell Set-Clipboard.
"""

import os
import subprocess
import sys

TARGET_DIRS = ["api", "ml", "web", "tests", "scripts", "docs", "reports"]
ROOT_FILES = ["README.md", "pytest.ini", "railway.json", ".gitignore"]
VALID_EXTS = {".py", ".html", ".json", ".md", ".ini", ".yaml", ".txt"}

SKIP_FILES = {
    "deep_guide.html",
    "ALL_RESPONSIBLE_CODE.txt",
    "ALL_CODEBASE_FILES.txt",
    "package-lock.json",
    "transcript.jsonl",
    "transcript_full.jsonl"
}

def collect_all_files():
    root_dir = os.path.abspath(".")
    files_to_read = []

    # 1. Root files
    for rf in ROOT_FILES:
        p = os.path.join(root_dir, rf)
        if os.path.exists(p) and os.path.isfile(p):
            files_to_read.append(rf)

    # 2. Walk target directories
    for tdir in TARGET_DIRS:
        abs_tdir = os.path.join(root_dir, tdir)
        if not os.path.exists(abs_tdir):
            continue
        for current_root, dirs, filenames in os.walk(abs_tdir):
            # Skip caches, logs, legacy_archive, and temp dirs
            if any(skip in current_root for skip in ["__pycache__", ".pytest_cache", ".git", "legacy_archive", ".system_generated"]):
                continue
            for f in sorted(filenames):
                if f in SKIP_FILES:
                    continue
                # Skip large cache work files in data/cache
                if f.startswith("work_W") and f.endswith(".json"):
                    continue
                ext = os.path.splitext(f)[1].lower()
                if ext in VALID_EXTS:
                    rel_p = os.path.relpath(os.path.join(current_root, f), root_dir)
                    rel_p_norm = rel_p.replace("\\", "/")
                    files_to_read.append(rel_p_norm)

    def sort_key(path):
        parts = path.split("/")
        category_order = {
            "README.md": 0, "pytest.ini": 1, "railway.json": 2, ".gitignore": 3,
            "web": 10,
            "api": 20,
            "ml": 30,
            "tests": 40,
            "scripts": 50,
            "reports": 60,
            "docs": 70
        }
        cat = parts[0]
        return (category_order.get(cat, 99), path)

    return sorted(list(set(files_to_read)), key=sort_key)

def main():
    root_dir = os.path.abspath(".")
    files = collect_all_files()
    
    collected_chunks = []
    
    sep = "=" * 85
    header = (
        f"{sep}\n"
        "  SCIGRAPH AI 2.0 — COMPLETE CONSOLIDATED CODEBASE MANIFEST\n"
        "  Includes 100% of Frontend (web/index.html), API, ML, Scientometrics, Tests & Scripts\n"
        f"{sep}\n\n"
    )
    collected_chunks.append(header)

    total_files = 0
    total_lines = 0
    total_bytes = 0
    manifest = []

    for rel_path in files:
        full_path = os.path.join(root_dir, rel_path)
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            num_lines = content.count("\n") + 1
            num_bytes = len(content.encode("utf-8"))

            banner = (
                "\n" + "=" * 85 + "\n"
                f"FILE: {rel_path} | {num_lines:,} lines | {num_bytes:,} bytes\n"
                + "=" * 85 + "\n\n"
            )
            collected_chunks.append(banner)
            collected_chunks.append(content)
            collected_chunks.append("\n\n")

            total_files += 1
            total_lines += num_lines
            total_bytes += num_bytes
            manifest.append((rel_path, num_lines, num_bytes))
        except Exception as exc:
            print(f"[ERROR] Could not read {rel_path}: {exc}", file=sys.stderr)

    full_payload = "".join(collected_chunks)

    # 1. Save to persistent file
    output_filepath = os.path.join(root_dir, "ALL_CODEBASE_FILES.txt")
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(full_payload)
    print(f"\n[SUCCESS] Successfully written {total_files} files to: {output_filepath}")

    # 2. Copy directly to Windows Clipboard using PowerShell Set-Clipboard
    print(f"Copying {len(full_payload):,} characters ({total_bytes:,} bytes, {total_lines:,} lines) to Windows Clipboard...")
    try:
        ps_cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"Get-Content -Raw -Encoding utf8 -Path '{output_filepath}' | Set-Clipboard"
        ]
        res = subprocess.run(ps_cmd, check=True)
        print("[SUCCESS] ALL code has been copied directly to your Windows Clipboard!")
        print("          You can now press Ctrl+V anywhere to paste the entire codebase!")
    except Exception as e:
        print(f"[WARNING] PowerShell Set-Clipboard failed: {e}. Trying clip.exe fallback...", file=sys.stderr)
        try:
            p = subprocess.Popen(["clip"], stdin=subprocess.PIPE, text=True, encoding="utf-8")
            p.communicate(input=full_payload)
            print("[SUCCESS] Copied to clipboard via clip.exe!")
        except Exception as e2:
            print(f"[ERROR] Clipboard copy failed: {e2}", file=sys.stderr)

    print("\n" + "=" * 85)
    print(f"CODEBASE MANIFEST SUMMARY: {total_files} Files Included (Literally Everything)")
    print("=" * 85)
    for pth, lns, bts in manifest:
        tag = "[FRONTEND]" if "web/index.html" in pth else ("[NEW]" if pth in ["scripts/update_ui.py", "tests/test_audit_verification.py", "reports/heldout_predictions_summary.json"] else "")
        print(f"  * {pth:<55} : {lns:>5,} lines ({bts:>8,} bytes) {tag}")
    print("-" * 85)
    print(f"  TOTAL CODEBASE: {total_files} files | {total_lines:,} lines | {total_bytes:,} bytes")
    print("=" * 85)

if __name__ == "__main__":
    main()
