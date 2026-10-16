"""Configuration management for SciGraph AI using Pydantic Settings.

Ensures strict zero-secret leakage, environment-driven configuration,
and single source of truth for paths, OpenAlex settings, and cutoff parameters.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application and Data Pipeline Settings."""

    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    CHECKPOINTS_DIR: Path = BASE_DIR / "ml" / "gnn" / "checkpoints"

    # OpenAlex Polite Pool Configuration
    # Uses OPENALEX_POLITE_EMAIL environment variable if available; fallback is an unprivileged academic alias
    OPENALEX_POLITE_EMAIL: str = "researcher@scigraph.org"
    OPENALEX_API_KEY: Optional[str] = None
    OPENALEX_BASE_URL: str = "https://api.openalex.org"

    # Scientometric Cutoff & Trajectory Parameters
    CUTOFF_OFFSET_YEARS: int = 2  # T_cutoff = pub_year + 2
    TRAJECTORY_HORIZON_YEARS: int = 5  # Evaluation horizon = pub_year + 5
    MIN_PUB_YEAR: int = 2010
    MAX_PUB_YEAR: int = 2020

    # Dataset Ingestion Parameters
    CORPUS_TARGET_SIZE: int = 20000
    OPENALEX_BATCH_SIZE: int = 200
    RATE_LIMIT_DELAY_SECONDS: float = 0.05

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
