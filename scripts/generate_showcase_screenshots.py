"""
Generates clean tab showcase HTML files and captures high-resolution screenshots of each tab.
"""

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
SCREENSHOTS_DIR = REPO_ROOT / "reports" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path(r"C:\Users\ASUS\.gemini\antigravity\brain\74a75f19-0cf8-4ad2-9529-0a8dec2e7c8e\screenshots")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

index_html = (WEB_DIR / "index.html").read_text(encoding="utf-8")

# Configure tabs
tabs = [
    ("01_live_prediction_dashboard.png", "predictor", 1600, 1100),
    ("02_heldout_verification_scoreboard.png", "verify-lab", 1600, 1200),
    ("03_accuracy_improvement_progression.png", "model-data", 1600, 1350),
    ("04_anti_cheating_future_poison_test.png", "anti-cheating", 1600, 1100),
]

for filename, tab_id, width, height in tabs:
    page_html = index_html
    # Deactivate default predictor tab
    page_html = page_html.replace('id="viewPredictor" class="view-section active"', 'id="viewPredictor" class="view-section"')
    page_html = page_html.replace('id="tabBtnPredictor" class="tab-btn active"', 'id="tabBtnPredictor" class="tab-btn"')
    
    # Activate target tab
    if tab_id == "predictor":
        page_html = page_html.replace('id="viewPredictor" class="view-section"', 'id="viewPredictor" class="view-section active"')
        page_html = page_html.replace('id="tabBtnPredictor" class="tab-btn"', 'id="tabBtnPredictor" class="tab-btn active"')
    elif tab_id == "verify-lab":
        page_html = page_html.replace('id="viewVerifyLab" class="view-section"', 'id="viewVerifyLab" class="view-section active"')
        page_html = page_html.replace('id="tabBtnVerifyLab" class="tab-btn"', 'id="tabBtnVerifyLab" class="tab-btn active"')
    elif tab_id == "model-data":
        page_html = page_html.replace('id="viewModelData" class="view-section"', 'id="viewModelData" class="view-section active"')
        page_html = page_html.replace('id="tabBtnModelData" class="tab-btn"', 'id="tabBtnModelData" class="tab-btn active"')
    elif tab_id == "anti-cheating":
        page_html = page_html.replace('id="viewAntiCheating" class="view-section"', 'id="viewAntiCheating" class="view-section active"')
        page_html = page_html.replace('id="tabBtnAntiCheating" class="tab-btn"', 'id="tabBtnAntiCheating" class="tab-btn active"')

    temp_html_path = WEB_DIR / f"temp_{tab_id}.html"
    temp_html_path.write_text(page_html, encoding="utf-8")
    
    out_img = SCREENSHOTS_DIR / filename
    url = f"http://localhost:8080/static/temp_{tab_id}.html"
    
    cmd = [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        f"--window-size={width},{height}",
        f"--virtual-time-budget=6000",
        f"--screenshot={out_img}",
        url
    ]
    print(f"Capturing {tab_id} -> {filename}...")
    subprocess.run(cmd, capture_output=True, text=True, timeout=25)
    
    if out_img.exists():
        print(f"  [SUCCESS] {filename} ({out_img.stat().st_size:,} bytes)")
        shutil.copy2(out_img, ARTIFACT_DIR / filename)
        print(f"  [COPIED] -> {ARTIFACT_DIR / filename}")
    
    # Clean up temp html
    if temp_html_path.exists():
        temp_html_path.unlink()

print("All screenshots generated and copied successfully.")
