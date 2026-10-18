import os
import sys

# Force unbuffered output so logs appear in real time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

print(f"Starting server from root: {root_dir}")

try:
    from api.main import app
    print("Loaded api.main:app successfully!")
except Exception as e:
    import traceback
    print("Error importing api.main:")
    traceback.print_exc()
    sys.exit(1)

import uvicorn

if __name__ == "__main__":
    print("Calling uvicorn.run on 127.0.0.1:8080...")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
