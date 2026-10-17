"""
Captures high-resolution screenshots of the live SciGraph AI system
demonstrating verified findings, benchmark metrics, progression ladder, and anti-cheating proofs.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

ARTIFACT_SCREENSHOTS = Path(r"C:\Users\ASUS\.gemini\antigravity\brain\74a75f19-0cf8-4ad2-9529-0a8dec2e7c8e\screenshots")

def capture_all():
    os.makedirs('reports/screenshots', exist_ok=True)
    ARTIFACT_SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    chrome = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
    
    targets = [
        ('reports/screenshots/01_live_prediction_dashboard.png', 'http://localhost:8080/?tab=predictor', 1600, 1100, 8000),
        ('reports/screenshots/02_heldout_verification_scoreboard.png', 'http://localhost:8080/?tab=verify-lab', 1600, 1200, 8000),
        ('reports/screenshots/03_accuracy_improvement_progression.png', 'http://localhost:8080/?tab=model-data', 1600, 1400, 8000),
        ('reports/screenshots/04_anti_cheating_future_poison_test.png', 'http://localhost:8080/?tab=anti-cheating', 1600, 1100, 8000),
        ('reports/screenshots/05_side_by_side_paper_comparison.png', 'http://localhost:8080/?tab=compare', 1600, 1100, 8000),
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
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        if os.path.exists(out_abs):
            size = os.path.getsize(out_abs)
            print(f'  [SUCCESS] Created {out_rel} ({size:,} bytes)')
            fname = Path(out_rel).name
            shutil.copy2(out_abs, ARTIFACT_SCREENSHOTS / fname)
            print(f'  [COPIED] -> {ARTIFACT_SCREENSHOTS / fname}')
        else:
            print(f'  [ERROR] Failed to create {out_rel}. Stderr: {res.stderr}')

if __name__ == '__main__':
    capture_all()
