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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.config import settings
from ml.inference_service import inference_service
from api.routers.paper import router as paper_router
from api.routers.verification import router as verify_router

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


@app.get("/health")
def health_check():
    """Service health and model availability status."""
    is_ready = inference_service.model is not None
    works_count = len(inference_service.df_works) if inference_service.df_works is not None else 0
    return {
        "status": "healthy" if is_ready else "degraded",
        "service": "SciGraph AI 2.0",
        "model_loaded": is_ready,
        "model_name": (inference_service.model_card or {}).get("model_name", "HeteroGraphSAGE"),
        "corpus_works_indexed": works_count,
        "polite_pool": settings.OPENALEX_POLITE_EMAIL
    }


# Serve Web Dashboard at root /
web_dir = Path(__file__).resolve().parent.parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    @app.get("/")
    def serve_dashboard():
        return FileResponse(web_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
