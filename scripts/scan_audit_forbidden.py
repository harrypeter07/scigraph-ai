import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

targets = [
    r"9,000",
    r"3,000\s+Papers",
    r"world's first",
    r"70%",
    r"Zero Temporal Leakage"
]

patterns = [re.compile(t, re.IGNORECASE) for t in targets]

dirs_to_check = [Path("web"), Path("README.md"), Path("docs")]

matches_found = []

def check_file(p: Path):
    if "legacy_v1" in p.parts:
        return
    if p.suffix in [".html", ".md", ".js", ".py", ".json", ".css", ".txt"]:
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            for pat, target_str in zip(patterns, targets):
                for match in pat.finditer(content):
                    start = max(0, match.start() - 40)
                    end = min(len(content), match.end() + 40)
                    snippet = content[start:end].replace("\n", " ")
                    matches_found.append((str(p), target_str, snippet))
        except Exception as e:
            pass

for item in dirs_to_check:
    if item.is_file():
        check_file(item)
    elif item.is_dir():
        for p in item.rglob("*"):
            if p.is_file():
                check_file(p)

print(f"Total forbidden occurrences found: {len(matches_found)}")
for f, t, snip in matches_found:
    print(f"  [{f}] [{t}]: ... {snip} ...")
