"""
Script to dynamically collect EVERY SINGLE source code file in the repository:
- All Frontend (web/index.html)
- All Backend & API (api/**/*.py)
- All Machine Learning & GNN (ml/**/*.py, ml/**/*.json)
- All Scientometrics Modules (ml/scientometrics/**/*.py)
- All Test Suites (tests/**/*.py)
- All Operational & Audit Scripts (scripts/**/*.py)
- All Scientific Reports & Documentation (reports/*.md, reports/*.json, docs/*, README.md, pytest.ini)

Saves the complete output to 'ALL_CODEBASE_FILES.txt' and copies it directly
to the Windows Clipboard.
"""

import os
import subprocess
import sys

# Target directory prefixes to include
TARGET_DIRS = ["api", "ml", "web", "tests", "scripts", "docs", "reports"]

# Root files to include
ROOT_FILES = ["README.md", "pytest.ini", "railway.json", ".gitignore"]

# Extensions to include
VALID_EXTS = {".py", ".html", ".json", ".md", ".ini", ".yaml"}

# Explicit skips (large auto-generated html artifacts or data dumps)
SKIP_FILES = {
    "deep_guide.html",
    "ALL_RESPONSIBLE_CODE.txt",
    "ALL_CODEBASE_FILES.txt",
    "package-lock.json"
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
            # Skip caches
            if any(skip in current_root for skip in ["__pycache__", ".pytest_cache", ".git"]):
                continue
            for f in sorted(filenames):
                if f in SKIP_FILES:
                    continue
                ext = os.path.splitext(f)[1].lower()
                if ext in VALID_EXTS:
                    rel_p = os.path.relpath(os.path.join(current_root, f), root_dir)
                    rel_p_norm = rel_p.replace("\\", "/")
                    files_to_read.append(rel_p_norm)

    # Sort files logically: root first, then api, ml, web, tests, scripts, reports, docs
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
    
    header = (
        "=" * 85 + "\n"
        "  SCIGRAPH AI 2.0 — COMPLETE CONSOLIDATED CODEBASE MANIFEST\n"
        "  Includes 100% of Frontend (web/index.html), API, ML, Scientometrics, Tests & Scripts\n"
        "=" * 85 + "\n\n"
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
        # Use PowerShell Set-Clipboard reading directly from the written file to avoid shell argument length limits
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
        is_frontend = "[FRONTEND]" if "web/index.html" in pth else ""
        print(f"  * {pth:<52} : {lns:>5,} lines ({bts:>8,} bytes) {is_frontend}")
    print("-" * 85)
    print(f"  TOTAL CODEBASE: {total_files} files | {total_lines:,} lines | {total_bytes:,} bytes")
    print("=" * 85)

if __name__ == "__main__":
    main()
