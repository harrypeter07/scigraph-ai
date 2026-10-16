"""
Captures high-resolution screenshots of the live SciGraph AI system
demonstrating verified findings, benchmark metrics, and production claims.
"""

import os
import subprocess
import time

def capture_all():
    os.makedirs('reports/screenshots', exist_ok=True)
    chrome = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
    
    targets = [
        ('reports/screenshots/01_live_prediction_dashboard.png', 'http://localhost:8080/', 1600, 1050, 7000),
        ('reports/screenshots/02_model_and_data_verification.png', 'http://localhost:8080/#model-data', 1600, 1050, 5000),
        ('reports/screenshots/03_paper_comparison.png', 'http://localhost:8080/#compare', 1600, 1050, 6000),
        ('reports/screenshots/04_fastapi_swagger_docs.png', 'http://localhost:8080/docs', 1600, 1050, 5000),
    ]
    
    for out_rel, url, w, h, budget in targets:
        out_abs = os.path.abspath(out_rel)
        cmd = [
            chrome,
            '--headless=new',
            '--disable-gpu',
            f'--window-size={w},{h}',
            f'--virtual-time-budget={budget}',
            f'--screenshot={out_abs}',
            url
        ]
        print(f'Capturing {url} -> {out_rel}...')
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if os.path.exists(out_abs):
            size = os.path.getsize(out_abs)
            print(f'  [SUCCESS] Created {out_rel} ({size:,} bytes)')
        else:
            print(f'  [ERROR] Failed to create {out_rel}. Stderr: {res.stderr}')

if __name__ == '__main__':
    capture_all()
