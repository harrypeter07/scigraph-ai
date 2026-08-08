# OpenAlex Acquisition Plan & Schema Specification

> **Phase**: Phase 1 (Data Acquisition Design & Schema Specification)  
> **Status**: Ready for Gate Review  
> **Target API**: OpenAlex REST API (`https://api.openalex.org/works`)  
> **Primary Objective**: Define exact field extraction, temporal leakage tags, trajectory reconstruction, concept filtering, sample size targets, and rate-limiting strategies prior to writing acquisition code.

---

## 1. Field Extraction & Tagging Catalog

Every field extracted from the OpenAlex Works API endpoint is cataloged below with its assigned **System Role** (`IDENTIFIER`, `FEATURE`, `GRAPH`, `TARGET-SOURCE`, `METADATA`), data type, and **Leakage Risk Rating** (`NONE`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).

| Field Name | Data Type | System Role | Leakage Risk | Feature / Graph / Target Usage | Description & Leakage Control |
|---|---|---|---|---|---|
| `id` | String (URI) | `IDENTIFIER` | **NONE** | Primary Key | Canonical OpenAlex Paper URI (e.g. `https://openalex.org/W2741809807`). |
| `doi` | String | `IDENTIFIER` | **NONE** | Metadata | Digital Object Identifier. |
| `title` | String | `FEATURE` | **NONE** | Text Feature | Paper title text used for title TF-IDF or text embedding generation. |
| `publication_year` | Integer | `METADATA` | **NONE** | Temporal Split / Cohort | Publication year ($Y_{\text{pub}}$). Used as temporal snapshot boundary $T_{\text{cutoff}} = Y_{\text{pub}}$. |
| `publication_date` | String (ISO) | `METADATA` | **NONE** | Snapshot Filter | Exact publication date (YYYY-MM-DD). |
| `abstract_inverted_index` | Dict[str, List[int]] | `FEATURE` | **NONE** | Text Feature | Inverted index used to reconstruct the original abstract text for semantic embeddings. |
| `referenced_works` | List[String] | `GRAPH` | **LOW** | Outgoing Edges | List of paper IDs cited by this paper. All referenced works were published $\le Y_{\text{pub}}$, thus non-leaking. |
| `referenced_works_count` | Integer | `FEATURE` | **NONE** | Tabular Feature | Total number of references cited by this paper. |
| `cited_by_count` | Integer | `TARGET-SOURCE` | **CRITICAL** | Raw Material ONLY | Current lifetime citation count as reported today. **NEVER** used directly as an input feature. |
| `counts_by_year` | List[Dict] | `TARGET-SOURCE` | **MEDIUM** | Trajectory / Labels | Array of `{"year": Y, "cited_by_count": C}` objects. Used to reconstruct annual citation trajectories. |
| `authorships` | List[Dict] | `GRAPH` | **LOW** | Authorship Edges | Contains `author.id`, `author.display_name`, `institutions.id`, `institutions.display_name`, `institutions.country_code`. |
| `concepts` | List[Dict] | `GRAPH` | **NONE** | Concept Edges | OpenAlex concept tags with confidence scores (`id`, `display_name`, `level`, `score`). |
| `primary_topic` | Dict | `METADATA` | **NONE** | Subfield Cohort | Primary subfield topic mapping used for cohort percentile normalization. |

---

## 2. Citation Trajectory Reconstruction via `counts_by_year`

To strictly eliminate temporal leakage, raw `cited_by_count` is rejected as an input feature. Instead, `counts_by_year` is parsed to reconstruct the exact historical timeline of citations:

### A. Historical Citation Count at $T_{\text{cutoff}}$ (Input Feature)
For a paper $p$ published in year $Y_{\text{pub}}$, the historical citation count available at cutoff time $T_{\text{cutoff}} = Y_{\text{pub}}$ is computed as:
$$\text{Citations}_{\le T_{\text{cutoff}}}(p) = \sum_{y \le Y_{\text{pub}}} \text{count\_in\_year}(p, y)$$
*(Note: For papers snapshotted at publication year $Y_{\text{pub}}$, this count is typically 0 or low initial citations in the publication year).*

### B. 5-Year Citation Accumulation Trajectory ($\Delta \text{Citations}_5$, Ground Truth Label Source)
The citation impact generated during the 5-year prediction horizon ($Y_{\text{pub}} + 1 \dots Y_{\text{pub}} + 5$) is calculated strictly as:
$$\Delta \text{Citations}_5(p) = \sum_{y = Y_{\text{pub}} + 1}^{Y_{\text{pub}} + 5} \text{count\_in\_year}(p, y)$$

If a paper was published in $Y_{\text{pub}} = 2018$, its 5-year prediction window spans $2019$ through $2023$. Any citations accrued in $2024$ or later are excluded from label computation.

---

## 3. Acquisition Query Specification & Filters

### AI/ML Concept & Subfield Filters
- **Primary Field Concept**: `C41008148` (Computer Science / Artificial Intelligence) or `C154945302` (Artificial Neural Networks / Machine Learning).
- **Publication Year Range**: $2012 \le Y_{\text{pub}} \le 2022$.
- **Document Type**: `type:article|conference-paper`.
- **Abstract Availability**: `has_abstract:true`.

### Target Sample Size & Phased Pull Strategy
1. **Initial Development Sample**: $5,000$ papers (used for rapid local pipeline validation and baseline training).
2. **Full Baseline Cohort**: $20,000$ papers across $2012–2022$.

---

## 4. Rate Limiting, Pagination & Caching Strategy

### Polite Pool & Rate Limiting
- **Polite Pool Headers**: Requests append `mailto` parameter sourced from `.env` (`OPENALEX_POLITE_EMAIL`).
- **Throttle Delay**: Enforced $0.1$ second inter-request delay (`rate_limit_delay_seconds: 0.1` in `configs/dataset.yaml`).
- **Retry Logic**: Exponential backoff on HTTP status codes 429, 500, 502, 503, 504 (up to 5 retries).

### Cursor Pagination
- Standard offset pagination is avoided to prevent deep-page performance degradation.
- Acquisition uses **Cursor Pagination** (`cursor=*` for initial request, updating `cursor=next_cursor` on subsequent responses until `next_cursor` is null).

### Resumable Local Raw Archive Caching
- Raw JSONL files are saved sequentially to `data/raw/openalex/openalex_ai_ml_batch_{batch_index}.jsonl`.
- Alongside raw data, a run log file `data/raw/openalex/ingestion_state.json` maintains:
  - Last successful cursor token.
  - Total records ingested to date.
  - Active query parameters & timestamp.
- **Idempotency**: If the script is interrupted, re-running `python -m ml.acquisition.openalex` resumes seamlessly from the saved cursor token without duplicate network requests or overwriting existing raw JSONL files.
