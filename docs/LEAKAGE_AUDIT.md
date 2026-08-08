# Temporal Leakage Audit & Mitigation Catalog

> **Core Research Contribution**: Temporal leakage occurs when information from after a target paper's observation cutoff ($T_{\text{cutoff}}$) is inadvertently included in feature extraction or graph topology creation.

---

## 1. Catalog of Identified Leakage Vectors

| Leakage Vector | Description | Severity | Resolution / Mitigation Strategy |
|---|---|---|---|
| **Raw `cited_by_count`** | OpenAlex reports current lifetime citation count. Using this directly leaks future citations. | **CRITICAL** | **FORBIDDEN** as a raw feature. Only used via `counts_by_year` to reconstruct citations accumulated up to $T_{\text{cutoff}}$. |
| **Future Citation Edges** | In graph construction, linking paper $A$ to paper $B$ when citation occurred at $t > T_{\text{cutoff}}$. | **CRITICAL** | Filter `paper_citations` edges to enforce $Y_{\text{citation}} \le T_{\text{cutoff}}$. |
| **Author Lifetime Metrics** | `author.works_count` or `author.cited_by_count` as reported today reflects future productivity. | **HIGH** | Reconstruct author features dynamically using only publications and citations published prior to $T_{\text{cutoff}}$. |
| **Future Institution Affiliations** | Using an author's present institution rather than their affiliation at paper publication date. | **MEDIUM** | Extract institution from historical paper `authorships.institutions` metadata. |
| **Random Data Splitting** | Performing a 70/15/15 random train/test split across mixed publication years. | **HIGH** | Enforce **Time-Consistent Splitting** by publication year ($Train \le 2018$, $Val = 2019$, $Test \ge 2020$). |

---

## 2. Automated Leakage Prevention Checks & Empirical Verification

The test suite enforces mandatory temporal assertions:
1. `test_leakage_audit.py`: Asserts that no feature column in `data/processed/` or `data/interim/` contains timestamps or citation increments recorded after $T_{\text{cutoff}}$.
2. `test_temporal_splits.py`: Asserts zero overlap in paper IDs between train, validation, and test splits, and verifies strict year monotonicity ($Y_{\text{train}} < Y_{\text{val}} < Y_{\text{test}}$).

### Phase 3 Verification Results
- **Sample Audited**: 50 OpenAlex AI/ML publications ($2012-2022$).
- **Interim Datasets**: Generated `papers.parquet`, `authors.parquet`, `institutions.parquet`, `authorships.parquet`, `paper_citations.parquet`, and `snapshotted_papers.parquet`.
- **Temporal Enforcement**: Verified `historical_citation_count_at_cutoff` strictly aggregates $y \le Y_{\text{pub}}$, and `delta_citations_5y` strictly aggregates $Y_{\text{pub}}+1 \dots Y_{\text{pub}}+5$.
- **Unit Test Status**: PASS (100% assertions satisfied).
