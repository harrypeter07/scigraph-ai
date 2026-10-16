"""
Script to collect all responsible core application files (~85%+ of codebase)
with relative paths and full source code, save to a consolidated file,
and copy the entire content directly to the Windows system clipboard.
"""

import os
import subprocess
import sys

# Priority list of core files representing the essential logic of SciGraph AI 2.0
RESPONSIBLE_FILES = [
    # 1. API & Backend Architecture
    "api/main.py",
    "api/config.py",
    "api/openalex_client.py",
    "api/routers/paper.py",
    "api/schemas/paper.py",

    # 2. Deep Learning & GNN Engine
    "ml/gnn/models/hetero_gnn.py",
    "ml/gnn/graph_builder.py",
    "ml/inference_service.py",
    "ml/features/extractor.py",
    "ml/baselines/benchmark_suite.py",
    "ml/gnn/checkpoints/MODEL_CARD.json",

    # 3. Scientometrics Modules
    "ml/scientometrics/venue_resolver.py",
    "ml/scientometrics/access_economics.py",
    "ml/scientometrics/lineage_engine.py",

    # 4. Interactive Frontend UI
    "web/index.html",

    # 5. Automated Test Suites (Verification)
    "tests/test_backend_api.py",
    "tests/test_phase2_scientometrics.py",
    "tests/test_phase3_model.py",

    # 6. Data Engineering & Scripts
    "scripts/compute_oa_advantage.py",
    "scripts/ingest_openalex_corpus.py",
    "scripts/ingest_scimago_and_core.py",
    "scripts/validate_dataset.py",
    "scripts/capture_all_screenshots.py",

    # 7. Core Documentation, Audits & Specifications
    "README.md",
    "AUDIT.md",
    "data/DATA_CARD.md",
    "reports/BENCHMARK_REPORT.md",
    "reports/OA_ADVANTAGE_ANALYSIS.md",
]

def main():
    root_dir = os.path.abspath(".")
    collected_chunks = []
    
    header = (
        "#" * 80 + "\n"
        "# SCIGRAPH AI 2.0 — CONSOLIDATED CORE APPLICATION CODEBASE MANIFEST\n"
        "# Generated for Code Review, Auditing, and Verification\n"
        "#" * 80 + "\n\n"
    )
    collected_chunks.append(header)
    
    total_files = 0
    total_lines = 0
    total_bytes = 0
    file_manifest = []

    for rel_path in RESPONSIBLE_FILES:
        full_path = os.path.normpath(os.path.join(root_dir, rel_path))
        if not os.path.exists(full_path):
            print(f"[WARNING] File not found: {rel_path}", file=sys.stderr)
            continue
            
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                
            num_lines = content.count("\n") + 1
            num_bytes = len(content.encode("utf-8"))
            
            file_banner = (
                "\n" + "=" * 80 + "\n"
                f"FILE: {rel_path} ({num_lines:,} lines, {num_bytes:,} bytes)\n"
                + "=" * 80 + "\n\n"
            )
            
            collected_chunks.append(file_banner)
            collected_chunks.append(content)
            collected_chunks.append("\n\n")
            
            total_files += 1
            total_lines += num_lines
            total_bytes += num_bytes
            file_manifest.append((rel_path, num_lines, num_bytes))
            
        except Exception as exc:
            print(f"[ERROR] Could not read {rel_path}: {exc}", file=sys.stderr)

    full_payload = "".join(collected_chunks)
    
    # 1. Save persistent backup copy
    backup_file = os.path.abspath("ALL_RESPONSIBLE_CODE.txt")
    with open(backup_file, "w", encoding="utf-8") as f:
        f.write(full_payload)
    print(f"[SUCCESS] Saved persistent backup to: {backup_file}")
    
    # 2. Pipe into Windows Clipboard using clip.exe
    print(f"Piping {len(full_payload):,} characters ({total_bytes:,} bytes) into Windows Clipboard...")
    p = subprocess.Popen(["clip"], stdin=subprocess.PIPE, text=True, encoding="utf-8")
    p.communicate(input=full_payload)
    
    if p.returncode == 0:
        print("[SUCCESS] All codebase content has been copied directly to your Windows Clipboard!")
    else:
        print(f"[WARNING] clip.exe returned code {p.returncode}. Attempting PowerShell fallback...")
        ps_cmd = ["powershell", "-Command", "Get-Content -Raw -Path 'ALL_RESPONSIBLE_CODE.txt' | Set-Clipboard"]
        subprocess.run(ps_cmd, check=True)
        print("[SUCCESS] PowerShell Set-Clipboard executed successfully!")

    print("\n" + "=" * 60)
    print(f"MANIFEST SUMMARY ({total_files} Primary Files Included):")
    print("=" * 60)
    for pth, lns, bts in file_manifest:
        print(f"  • {pth:<42} : {lns:>5,} lines ({bts:>7,} bytes)")
    print("-" * 60)
    print(f"  TOTAL: {total_lines:,} lines | {total_bytes:,} bytes | {len(full_payload):,} chars")
    print("=" * 60)

if __name__ == "__main__":
    main()
