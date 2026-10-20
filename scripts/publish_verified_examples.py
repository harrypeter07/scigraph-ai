"""Select explicitly curated successes by checking actual serving API responses."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def publish(base_url):
    import joblib
    checkpoint = ROOT / 'ml/gnn/checkpoints/forecast_v3.joblib'
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    session = requests.Session()
    evidence = session.get(base_url + '/api/v1/papers/forecast-evidence', timeout=30)
    evidence.raise_for_status()
    info = evidence.json()
    if not info.get('serving', {}).get('verified') or info['forecast_bundle_sha256'] != checksum:
        raise RuntimeError('API is not serving the evaluated checkpoint')
    rows = [r for r in joblib.load(checkpoint)['evaluation_reference'] if r['publication_year'] == 2020]
    random.Random(42).shuffle(rows)
    selected = []
    for tier in range(3):
        candidates = [r for r in rows if r['impact_class'] == tier and r['primary_field_name'] in ['Computer Science', 'Engineering', 'Physics and Astronomy']]
        count = 0
        for row in candidates[:60]:
            wid = row['work_id'].split('/')[-1]
            response = session.get(base_url + '/api/v1/papers/predict/' + wid, params={'mode':'M'}, timeout=60)
            if not response.ok:
                print('Skipping unavailable record', wid, response.status_code)
                continue
            data = response.json()
            bt = data.get('backtest') or {}
            if data.get('split') != 'test' or not bt.get('has_ground_truth') or bt.get('is_correct_class') is not True:
                continue
            selected.append({'paper_id':wid, 'title':data['title'], 'publication_year':2020,
                             'field':data.get('primary_field'), 'predicted_label':data['predicted_label'],
                             'actual_label':bt['actual_label'], 'actual_citations_window':bt['actual_citations_window'],
                             'verified_correct':True})
            print('Verified', wid, data['predicted_label'])
            count += 1
            if count == 2:
                break
        if count != 2:
            raise RuntimeError('Not enough serving-verified examples; keep previous manifest')
    payload = {'forecast_bundle_sha256':checksum, 'selection':'Curated correct predictions; not a representative accuracy sample.', 'examples':selected}
    path = ROOT / 'web/data/verified_examples.json'
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8081')
    publish(parser.parse_args().base_url.rstrip('/'))
