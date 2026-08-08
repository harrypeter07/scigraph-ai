"""FastAPI Entry Point for SciGraph AI Application Server."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers.paper import router as paper_router

app = FastAPI(
    title="SciGraph AI REST Service",
    description="API for Heterogeneous GNN Citation Trajectory & Impact Predictions",
    version="1.0.0"
)

# Enable CORS for Next.js research dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(paper_router)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "SciGraph AI Service"}
