"""Build a portable Colab notebook + explicit source/data whitelist.

Never packages .env, API keys, raw API dumps, caches, or unrelated user files.
"""
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return dict(cell_type="markdown", metadata={}, source=text.splitlines(keepends=True))


def code(text):
    return dict(cell_type="code", execution_count=None, metadata={}, outputs=[], source=text.splitlines(keepends=True))


def package():
    sources = ["ml/features/forecast.py", "ml/gnn/citation_forecast.py", "ml/gemini_service.py",
               "scripts/train_forecast.py", "scripts/train_citation_forecast.py",
               "scripts/acquire_forecast_corpus.py", "scripts/enrich_citation_neighbors.py", "scripts/activate_forecast.py",
               "tests/test_forecast_contract.py"]
    kit = ROOT / "reports/forecast_training_kit.zip"
    with zipfile.ZipFile(kit, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for package_name in ["ml", "ml/features", "ml/gnn", "scripts"]:
            archive.writestr(f"{package_name}/__init__.py", "")
        for relative in sources:
            archive.write(ROOT / relative, relative)
        for directory in ["data", "data/expanded_forecast"]:
            for name in ["works.parquet", "counts_by_year.parquet", "citation_edges.parquet", "acquisition_audit.json",
                         "neighbor_works.parquet", "neighbor_counts_by_year.parquet", "neighbor_edges.parquet", "neighbor_audit.json"]:
                path = ROOT / directory / name
                if path.exists():
                    archive.write(path, f"{directory}/{name}")
    cells = [markdown("""# SciGraph: reproducible citation-impact training

Upload `forecast_training_kit.zip` from this project. It contains source and public corpus tables, not credentials.

This notebook measures results; it does not promise an accuracy. The original corpus and expanded random corpus have different sampling distributions. Report each separately. Citation impact is not scientific quality. Cohort labels and early-count normalization are sample-relative, with explicit handling of ties.

The existing 2018–2020 benchmark has informed development. Repeated Colab runs do not turn it into fresh confirmation. Select models on the 2017 selection partition, calibrate on a separate 2017 partition, and reserve any future confirmation dataset before running more experiments.

GPU is optional for tabular models. The citation GNN automatically uses a GPU when available.
"""), code("""import sys, subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'scikit-learn==1.7.2', 'pandas', 'pyarrow', 'numpy', 'scipy', 'joblib', 'lightgbm', 'pytest', 'requests'], check=True)
try:
    import torch
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch'], check=True)
    import torch
print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')
"""), code("""from google.colab import files
from pathlib import Path
import zipfile, os
uploaded = files.upload()  # Choose forecast_training_kit.zip
kit_path = next(Path(name) for name in uploaded if name.endswith('.zip'))
project = Path('/content/scigraph_forecast').resolve()
project.mkdir(exist_ok=True)
with zipfile.ZipFile(kit_path) as archive:
    for item in archive.infolist():
        target = (project/item.filename).resolve()
        if not target.is_relative_to(project):
            raise ValueError('Unsafe archive path')
    archive.extractall(project)
os.chdir(project)
sys.path.insert(0, str(project))
subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_forecast_contract.py', '-q'], check=True)
"""), markdown("""## Choose the corpus

The expanded corpus is included when available. Its original-corpus ID overlap is zero. It samples scholarly works across five fields without filtering for abstracts or existing citations. Papers with unreconciled citation histories are excluded and counted in the acquisition audit. Unknown author metadata is disabled during fitting if coverage is insufficient.

Leave acquisition disabled to reproduce the packaged data. If acquiring a new corpus, register its final evaluation set before inspecting scores. Optional API credentials remain in the notebook runtime and are never exported.
"""), code("""RUN_ACQUISITION = False
if RUN_ACQUISITION:
    # Optional: use getpass and os.environ['OPENALEX_API_KEY'] for your own key.
    subprocess.run([sys.executable, 'scripts/acquire_forecast_corpus.py', '--per-cohort', '1200'], check=True)
data_dir = Path('data/expanded_forecast') if Path('data/expanded_forecast/works.parquet').exists() else Path('data')
print('Chosen corpus:', data_dir)
"""), code("""TAB_OUT = Path('reports/colab_forecast')
subprocess.run([sys.executable, 'scripts/train_forecast.py', '--data-dir', str(data_dir), '--output-dir', str(TAB_OUT)], check=True)
import json
tab_report = json.loads((TAB_OUT/'evaluation.json').read_text())
score = tab_report['development_test']
print('Retrospective pooled accuracy:', score['accuracy'], 'CI:', score['accuracy_ci95'])
strict = tab_report['strict_temporal_test']
print('Label-maturity eligible holdout:', strict['publication_years'], 'N=', strict['n'])
print('Eligible accuracy:', strict['accuracy'], 'CI:', strict['accuracy_ci95'], 'Macro-F1:', strict['macro_f1'])
print('Macro-F1:', score['macro_f1'], 'Majority baseline:', score['majority_baseline_accuracy'])
print('Field breakdown:', {k:v['accuracy'] for k,v in score['by_field'].items()})
print('Class precision / recall:', score['per_class'])
print('Above 80% on this corpus:', score['above_80_percent'])
"""), markdown("""## Test the actual citation GNN

This is a one-hop relational GraphSAGE model with distinct incoming/outgoing citation message functions. It uses actual neighbor papers and target-cutoff features. Missing graph neighborhoods stay empty. Three seeds are averaged. Epochs and fusion weights are selected on validation only; temperature is fitted on the calibration partition.

The removed-edge result is a sensitivity test. The separately retrained no-edge control also tests whether improvement is due to graph information rather than merely neural capacity. Do not force a positive graph fusion weight.
"""), code("""ENRICH_GRAPH = False  # Packaged neighbor tables are used automatically when present.
if ENRICH_GRAPH:
    subprocess.run([sys.executable, 'scripts/enrich_citation_neighbors.py', '--data-dir', str(data_dir), '--maximum', '40000'], check=True)
GRAPH_OUT = Path('reports/colab_citation')
subprocess.run([sys.executable, 'scripts/train_citation_forecast.py', '--data-dir', str(data_dir), '--bundle', str(TAB_OUT/'forecast_v3.joblib'), '--output-dir', str(GRAPH_OUT), '--epochs', '80'], check=True)
graph_report = json.loads((GRAPH_OUT/'graph_evaluation.json').read_text())
print('Validation-selected graph weight:', graph_report['selected_graph_weight'])
print('Tabular:', graph_report['development_tabular']['accuracy'])
print('Graph/fusion:', graph_report['development_fusion']['accuracy'])
print('Removed messages:', graph_report['development_removed_edges']['accuracy'])
print('Paired lift CI:', graph_report['paired_bootstrap_lift_ci95'])
print('Coverage:', graph_report['graph_coverage'])
"""), code("""CONTROL_OUT = Path('reports/colab_no_edges')
subprocess.run([sys.executable, 'scripts/train_citation_forecast.py', '--data-dir', str(data_dir), '--bundle', str(TAB_OUT/'forecast_v3.joblib'), '--output-dir', str(CONTROL_OUT), '--epochs', '80', '--without-edges'], check=True)
control = json.loads((CONTROL_OUT/'graph_evaluation.json').read_text())
print('Root-only neural control:', control['development_gnn']['accuracy'])
"""), markdown("""## Export measured artifacts

Download the ZIP and bring it back to the project. It contains the matching tabular and graph checkpoints plus reports and runtime versions. In the existing repository, install with `python scripts/activate_forecast.py PATH/forecast_v3.joblib --graph PATH/citation_gnn_experiment.joblib`. A local installation creates backups; it does not publish to Railway. Graph serving requires PyTorch. Keep scikit-learn at 1.7.2.
"""), code("""import importlib.metadata
versions = {name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','joblib','torch']}
Path('runtime_versions.json').write_text(json.dumps(versions, indent=2))
export = Path('trained_forecast_bundle.zip')
with zipfile.ZipFile(export, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for directory in [TAB_OUT, GRAPH_OUT, CONTROL_OUT]:
        for path in directory.iterdir():
            if path.suffix in ['.joblib','.json']:
                archive.write(path, f'{directory.name}/{path.name}')
    archive.write('runtime_versions.json')
files.download(str(export))
""")]
    notebook = dict(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, nbformat=4, nbformat_minor=5)
    destination = ROOT / "notebooks/scigraph_verified_forecast_colab.ipynb"
    destination.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(destination)
    print(kit, kit.stat().st_size, "bytes")


if __name__ == "__main__":
    package()
