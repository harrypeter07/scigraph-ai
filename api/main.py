"""FastAPI Entry Point for SciGraph AI Application Server.

Enforces:
- Rule R2: Loud failure at startup if ML checkpoint or model card is missing/corrupt.
- Single source of truth configuration via pydantic-settings.
- Clean REST routes with provenance-aware responses.
"""

import os
import warnings
import logging
from contextlib import asynccontextmanager
from pathlib import Path

# Suppress PyTorch internal JIT deprecation warnings emitted by torch_geometric
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.jit.script.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

import hashlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from api.config import settings
from ml.inference_service import inference_service
from api.routers.paper import router as paper_router
from api.routers.verification import router as verify_router
from api.routers.topic import router as topic_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("scigraph.api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager strictly enforcing Rule R2 (Loud failures at boot)."""
    logger.info("Starting SciGraph AI Application Server...")
    logger.info(f"OpenAlex polite pool email: {settings.OPENALEX_POLITE_EMAIL}")
    
    # Load model weights, reference tables, and model card
    try:
        inference_service.load()
        logger.info("Model and reference scientometric tables loaded successfully.")
    except Exception as exc:
        logger.critical(f"FATAL BOOT ERROR: {exc}", exc_info=True)
        raise RuntimeError(f"Server startup aborted: {exc}") from exc

    yield

    logger.info("Shutting down SciGraph AI Application Server.")


app = FastAPI(
    title="SciGraph AI REST Service",
    description="Scientific 5-Year Citation Trajectory Impact Prediction via Heterogeneous GNN",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers for both singular and plural API paths
app.include_router(paper_router, prefix="/api/v1/papers")
app.include_router(paper_router, prefix="/api/v1/paper")
app.include_router(verify_router, prefix="/api/v1/verify")
app.include_router(topic_router)


_cached_splits = None

def get_splits_data():
    global _cached_splits
    if _cached_splits is not None:
        return _cached_splits
    import pandas as pd
    base_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    train_path = base_dir / "train.parquet"
    val_path = base_dir / "val.parquet"
    test_path = base_dir / "test.parquet"

    train_df = pd.read_parquet(train_path) if train_path.exists() else None
    val_df = pd.read_parquet(val_path) if val_path.exists() else None
    test_df = pd.read_parquet(test_path) if test_path.exists() else None

    train_total = len(train_df) if train_df is not None else 0
    train_honest = int((train_df["publication_year"] >= 2012).sum()) if train_df is not None else 0
    train_pre2012_excluded = train_total - train_honest
    val_count = len(val_df) if val_df is not None else 0
    test_count = len(test_df) if test_df is not None else 0

    _cached_splits = {
        "train_2012_2016": train_honest,
        "train_honest": train_honest,
        "train_pre2012_excluded": train_pre2012_excluded,
        "train_parquet_total": train_total,
        "val_2017": val_count,
        "val": val_count,
        "test_2018_2020": test_count,
        "test": test_count,
        "total_active_benchmark": train_honest + val_count + test_count,
        "total_corpus": train_total + val_count + test_count,
        "years": {
            "train": "2012–2016",
            "val": "2017",
            "test": "2018–2020"
        }
    }
    return _cached_splits


@app.get("/api/v1/meta/splits")
def get_meta_splits():
    """Returns dataset split counts computed directly from processed parquet tables and masks."""
    return get_splits_data()


@app.get("/api/v1/meta/numbers")
def get_numbers_registry():
    """Returns the unified quantitative numbers registry from docs/NUMBERS.json."""
    from scripts.make_numbers import collect_numbers
    return collect_numbers()


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="6" fill="#246B50"/>'
        '<text x="16" y="22" text-anchor="middle" fill="#fff" font-size="16" font-family="Arial">S</text>'
        "</svg>"
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/health")
def health_check():
    """Service health and model availability status."""
    primary_ready = getattr(inference_service, "primary_model", None) is not None
    is_ready = primary_ready or (inference_service.model is not None)
    works_count = len(inference_service.df_works) if inference_service.df_works is not None else 0

    manifest_path = Path(__file__).resolve().parent.parent / "data" / "MANIFEST.sha256"
    manifest_sha = None
    if manifest_path.exists():
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    return {
        "status": "healthy" if primary_ready else "degraded",
        "service": "SciGraph AI 2.0",
        "deployed_commit": "deploy-v7-torch-guard",
        "model_loaded": primary_ready,
        "primary_model_loaded": primary_ready,
        "gnn_model_loaded": inference_service.model is not None,
        "model_name": "M1_LogisticRegression",
        "corpus_works_indexed": works_count,
        "polite_pool": settings.OPENALEX_POLITE_EMAIL,
        "data_snapshot": {
            "openalex_retrieved_utc": "2026-10-15T08:08:00Z",
            "manifest_sha256": manifest_sha,
            "manifest_path": "data/MANIFEST.sha256"
        }
    }


# Serve Web Dashboard and Static Assets
web_dir = Path(__file__).resolve().parent.parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")
    if (web_dir / "css").exists():
        app.mount("/css", StaticFiles(directory=str(web_dir / "css")), name="css")
    if (web_dir / "js").exists():
        app.mount("/js", StaticFiles(directory=str(web_dir / "js")), name="js")
    if (web_dir / "views").exists():
        app.mount("/views", StaticFiles(directory=str(web_dir / "views")), name="views")

    @app.get("/")
    def serve_dashboard():
        return FileResponse(web_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
