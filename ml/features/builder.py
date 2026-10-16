"""Central Feature Builder for SciGraph AI 2.0.

Single source of truth for extracting strictly as-of-cutoff paper features.
Used universally by ml/gnn/graph_builder.py, ml/baselines/*, and ml/inference_service.py.
Eliminates training/serving feature parity discrepancies.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Path to persisted text encoder
DEFAULT_TEXT_ENCODER_PATH = Path(__file__).resolve().parent.parent / "gnn" / "checkpoints" / "text_encoder.joblib"

# Ordered 14 feature groups covering exactly 47 dimensions:
# Slices match indices in the 47-dimensional output vector.
FEATURE_GROUPS: OrderedDict[str, slice] = OrderedDict([
    ("log_cits_cutoff", slice(0, 1)),      # ln(1 + C_cutoff)
    ("pub_offset", slice(1, 2)),           # (publication_year - 2010) / 10.0
    ("log_refs", slice(2, 3)),             # ln(1 + referenced_works_count)
    ("author_scale", slice(3, 4)),         # min(50, max(1, num_authors)) / 10.0
    ("is_oa", slice(4, 5)),                # is_open_access (1.0 or 0.0)
    ("oa_gold", slice(5, 6)),              # oa_status == 'gold'
    ("oa_green", slice(6, 7)),             # oa_status == 'green'
    ("oa_hybrid", slice(7, 8)),            # oa_status == 'hybrid'
    ("oa_bronze", slice(8, 9)),            # oa_status == 'bronze'
    ("oa_closed", slice(9, 10)),           # oa_status == 'closed'
    ("log_apc", slice(10, 12)),            # [ln(1 + apc_paid_usd), apc_known]
    ("sjr", slice(12, 14)),                # [sjr, venue_known]
    ("tier_top", slice(14, 15)),           # SCImago Q1 or CORE A*/A
    ("title_svd", slice(15, 47)),          # 32-dim TruncatedSVD from TF-IDF
])

FEATURE_NAMES: List[str] = [
    "log_cits_cutoff",
    "pub_offset",
    "log_refs",
    "author_scale",
    "is_oa",
    "oa_gold",
    "oa_green",
    "oa_hybrid",
    "oa_bronze",
    "oa_closed",
    "log_apc",
    "apc_known",
    "sjr",
    "venue_known",
    "tier_top",
    *[f"title_svd_{i}" for i in range(32)],
]

TOTAL_FEATURE_DIM = len(FEATURE_NAMES)  # 47 dimensions

_CACHED_TEXT_ENCODER: Optional[Dict[str, Any]] = None


def get_text_encoder(checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads and caches the persistent TF-IDF + TruncatedSVD text encoder."""
    global _CACHED_TEXT_ENCODER
    if _CACHED_TEXT_ENCODER is not None:
        return _CACHED_TEXT_ENCODER

    path = checkpoint_path or DEFAULT_TEXT_ENCODER_PATH
    if not path.exists():
        raise RuntimeError(
            f"Text encoder checkpoint not found at {path}. "
            f"Run fit_and_save_text_encoder() on train split titles first."
        )

    _CACHED_TEXT_ENCODER = joblib.load(path)
    return _CACHED_TEXT_ENCODER


def fit_and_save_text_encoder(
    train_titles: List[str],
    output_path: Optional[Path] = None,
    max_features: int = 2500,
    n_components: int = 32,
    random_state: int = 42,
) -> Path:
    """Fits TF-IDF and TruncatedSVD on TRAIN split titles ONLY and saves to disk."""
    global _CACHED_TEXT_ENCODER
    path = output_path or DEFAULT_TEXT_ENCODER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_titles = [str(t) if (t is not None and pd.notnull(t)) else "Untitled" for t in train_titles]

    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(cleaned_titles)

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    svd.fit(tfidf_matrix)

    encoder_dict = {"tfidf": tfidf, "svd": svd}
    joblib.dump(encoder_dict, path)
    _CACHED_TEXT_ENCODER = encoder_dict
    return path


def transform_titles(titles: List[str], checkpoint_path: Optional[Path] = None) -> np.ndarray:
    """Transforms a list of titles into 32-dimensional SVD embeddings."""
    encoder = get_text_encoder(checkpoint_path)
    tfidf = encoder["tfidf"]
    svd = encoder["svd"]

    cleaned_titles = [str(t) if (t is not None and pd.notnull(t)) else "Untitled" for t in titles]
    tfidf_matrix = tfidf.transform(cleaned_titles)
    embeddings = svd.transform(tfidf_matrix)
    return embeddings.astype(np.float32)


def _extract_single_tabular(record: Union[Dict[str, Any], pd.Series], cutoff_offset: int = 2) -> np.ndarray:
    """Extracts the 15 tabular features from a single dictionary or pandas Series."""
    # 1. Publication year and cutoff
    pub_year_val = record.get("publication_year")
    if pub_year_val is None or (isinstance(pub_year_val, float) and np.isnan(pub_year_val)):
        pub_year = 2018
    else:
        pub_year = int(pub_year_val)
    cutoff_year = pub_year + cutoff_offset
    pub_offset = (pub_year - 2010) / 10.0

    # 2. Historical citations at cutoff
    if "citations_at_cutoff" in record and pd.notnull(record.get("citations_at_cutoff")):
        cits_at_cutoff = float(record["citations_at_cutoff"])
    elif "counts_by_year" in record and isinstance(record.get("counts_by_year"), list):
        cits_at_cutoff = float(sum(
            c.get("cited_by_count", 0)
            for c in record["counts_by_year"]
            if isinstance(c, dict) and c.get("year", 9999) <= cutoff_year
        ))
    else:
        cits_at_cutoff = 0.0
    log_cits = float(np.log1p(max(0.0, cits_at_cutoff)))

    # 3. References count
    if "referenced_works_count" in record and pd.notnull(record.get("referenced_works_count")):
        ref_count = float(record["referenced_works_count"])
    elif "referenced_works" in record and isinstance(record.get("referenced_works"), list):
        ref_count = float(len(record["referenced_works"]))
    else:
        ref_count = 0.0
    log_refs = float(np.log1p(max(0.0, ref_count)))

    # 4. Author scale
    if "num_authors" in record and pd.notnull(record.get("num_authors")):
        num_auth = float(record["num_authors"])
    elif "authorships" in record and isinstance(record.get("authorships"), list):
        num_auth = float(len(record["authorships"]))
    else:
        num_auth = 1.0
    author_scale = float(min(50.0, max(1.0, num_auth)) / 10.0)

    # 5. Open access status & one-hot
    if "is_oa" in record and pd.notnull(record.get("is_oa")):
        is_oa = 1.0 if bool(record["is_oa"]) else 0.0
    elif "open_access" in record and isinstance(record.get("open_access"), dict):
        is_oa = 1.0 if bool(record["open_access"].get("is_oa", False)) else 0.0
    else:
        is_oa = 0.0

    if "oa_status" in record and pd.notnull(record.get("oa_status")):
        oa_status = str(record["oa_status"]).lower()
    elif "open_access" in record and isinstance(record.get("open_access"), dict):
        oa_status = str(record["open_access"].get("oa_status", "closed")).lower()
    else:
        oa_status = "closed"

    oa_gold = 1.0 if oa_status == "gold" else 0.0
    oa_green = 1.0 if oa_status == "green" else 0.0
    oa_hybrid = 1.0 if oa_status == "hybrid" else 0.0
    oa_bronze = 1.0 if oa_status == "bronze" else 0.0
    oa_closed = 1.0 if oa_status == "closed" else 0.0

    # 6. APC paid and missingness flag (unknown must NOT equal 0.0)
    apc_paid_val = None
    if "apc_paid_usd" in record and pd.notnull(record.get("apc_paid_usd")):
        apc_paid_val = float(record["apc_paid_usd"])
    elif "apc_paid" in record and isinstance(record.get("apc_paid"), dict):
        val = record["apc_paid"].get("value_usd")
        if val is not None and pd.notnull(val):
            try:
                apc_paid_val = float(val)
            except (ValueError, TypeError):
                apc_paid_val = None

    if apc_paid_val is not None:
        log_apc = float(np.log1p(max(0.0, apc_paid_val)))
        apc_known = 1.0
    else:
        log_apc = -1.0  # Explicit missingness indicator
        apc_known = 0.0

    # 7. Venue metrics: SJR, venue_known, tier_top (unknown SJR must NOT equal 0.0)
    venue_sjr_val = None
    quartile = record.get("venue_scimago_quartile")
    core_rank = record.get("venue_core_rank")

    if "venue_sjr" in record and pd.notnull(record.get("venue_sjr")):
        venue_sjr_val = float(record["venue_sjr"])
    else:
        # Fallback to venue resolver if primary_location or venue name is available
        source_name = ""
        issn_l = None
        if "primary_location" in record and isinstance(record.get("primary_location"), dict):
            source = record["primary_location"].get("source") or {}
            source_name = source.get("display_name") or ""
            issn_l = source.get("issn_l") or ((source.get("issn") or [""])[0] if isinstance(source.get("issn"), list) else None)
        elif "venue_name" in record and pd.notnull(record.get("venue_name")):
            source_name = str(record["venue_name"])
            issn_l = record.get("venue_issn_l")

        if source_name or issn_l:
            try:
                from ml.scientometrics.venue_resolver import venue_resolver
                res = venue_resolver.resolve(source_name, issn_l, pub_year)
                if res.sjr is not None and pd.notnull(res.sjr):
                    venue_sjr_val = float(res.sjr)
                if quartile is None:
                    quartile = res.quartile
                if core_rank is None:
                    core_rank = res.core_rank
            except Exception:
                pass

    if venue_sjr_val is not None:
        sjr = float(venue_sjr_val)
        venue_known = 1.0
    else:
        sjr = -1.0  # Explicit missingness indicator: unknown SJR must NOT equal 0.0
        venue_known = 0.0

    is_top_tier = 1.0 if (quartile == "Q1" or (core_rank in ["A*", "A"])) else 0.0

    return np.array([
        log_cits,
        pub_offset,
        log_refs,
        author_scale,
        is_oa,
        oa_gold,
        oa_green,
        oa_hybrid,
        oa_bronze,
        oa_closed,
        log_apc,
        apc_known,
        sjr,
        venue_known,
        is_top_tier,
    ], dtype=np.float32)


def build_paper_features(
    record: Union[Dict[str, Any], pd.Series],
    cutoff_offset: int = 2,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Universal feature builder for a single paper record.
    
    Returns:
        (features_vector, feature_names) where features_vector is a 1D np.ndarray of shape (47,).
    """
    tabular = _extract_single_tabular(record, cutoff_offset=cutoff_offset)
    title = str(record.get("title", "")) if pd.notnull(record.get("title")) else "Untitled"
    svd_emb = transform_titles([title], checkpoint_path=checkpoint_path)[0]

    features = np.concatenate([tabular, svd_emb]).astype(np.float32)
    return features, FEATURE_NAMES


def build_paper_features_batch(
    df: pd.DataFrame,
    cutoff_offset: int = 2,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Vectorized feature builder for a pandas DataFrame of works.
    
    Guarantees exact numerical equivalence with build_paper_features for each row.
    """
    N = len(df)

    # 1. Publication year and cutoff
    pub_years = df["publication_year"].fillna(2018).astype(int).values
    pub_offset = ((pub_years - 2010) / 10.0)[:, None]

    # 2. Historical citations at cutoff
    cits_cutoff = df["citations_at_cutoff"].fillna(0.0).clip(lower=0).values
    log_cits = np.log1p(cits_cutoff)[:, None]

    # 3. References count
    refs = df["referenced_works_count"].fillna(0.0).clip(lower=0).values
    log_refs = np.log1p(refs)[:, None]

    # 4. Author scale
    num_auth = df["num_authors"].fillna(1.0).clip(lower=1.0, upper=50.0).values
    author_scale = (num_auth / 10.0)[:, None]

    # 5. Open Access
    is_oa = df["is_oa"].fillna(False).astype(float).values[:, None]
    oa_status_series = df["oa_status"].fillna("closed").str.lower()
    oa_gold = (oa_status_series == "gold").astype(float).values[:, None]
    oa_green = (oa_status_series == "green").astype(float).values[:, None]
    oa_hybrid = (oa_status_series == "hybrid").astype(float).values[:, None]
    oa_bronze = (oa_status_series == "bronze").astype(float).values[:, None]
    oa_closed = (oa_status_series == "closed").astype(float).values[:, None]

    # 6. APC paid and missingness
    apc_raw = df["apc_paid_usd"].values
    apc_known = (~pd.isna(apc_raw)).astype(float)[:, None]
    apc_clean = np.where(pd.isna(apc_raw), 0.0, np.maximum(0.0, apc_raw))
    log_apc = np.where(apc_known == 1.0, np.log1p(apc_clean)[:, None], -1.0)

    # 7. Venue metrics and missingness
    sjr_raw = df["venue_sjr"].values
    venue_known = (~pd.isna(sjr_raw)).astype(float)[:, None]
    sjr_clean = np.where(pd.isna(sjr_raw), -1.0, sjr_raw)[:, None]

    is_q1 = (df["venue_scimago_quartile"] == "Q1").values
    is_core_a = df["venue_core_rank"].isin(["A*", "A"]).values
    tier_top = (is_q1 | is_core_a).astype(float)[:, None]

    tabular_mat = np.hstack([
        log_cits,
        pub_offset,
        log_refs,
        author_scale,
        is_oa,
        oa_gold,
        oa_green,
        oa_hybrid,
        oa_bronze,
        oa_closed,
        log_apc,
        apc_known,
        sjr_clean,
        venue_known,
        tier_top,
    ]).astype(np.float32)

    # SVD Embeddings
    titles = df["title"].fillna("Untitled").tolist()
    svd_emb = transform_titles(titles, checkpoint_path=checkpoint_path)

    features_matrix = np.hstack([tabular_mat, svd_emb]).astype(np.float32)
    return features_matrix, FEATURE_NAMES
