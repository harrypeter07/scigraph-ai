"""FastAPI Entry Point for SciGraph AI Application Server."""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routers.paper import router as paper_router
from api.routers.evidence import router as evidence_router

app = FastAPI(
    title="SciGraph AI REST Service",
    description="API for Heterogeneous GNN Citation Trajectory & Impact Predictions",
    version="1.0.0"
)

# Enable CORS for Next.js / Web research dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(paper_router)
app.include_router(evidence_router)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "SciGraph AI Service"}


# Serve Web Dashboard at root /
web_dir = "web"
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/")
    def serve_dashboard():
        return FileResponse(os.path.join(web_dir, "index.html"))
