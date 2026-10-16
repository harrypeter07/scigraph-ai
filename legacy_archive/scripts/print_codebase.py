import sys

files = [
    "api/schemas/paper.py",
    "ml/gnn/models/graphsage.py",
    "tests/test_scientometrics.py",
    "api/routers/paper.py"
]

for filepath in files:
    sep = "=" * 80
    print("\n" + sep)
    print(f"FILE: {filepath}")
    print(sep + "\n")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            print(f.read())
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
