# Data Card: Authentic OpenAlex Multi-Disciplinary Scientometric Corpus

## 1. Dataset Summary

- **Repository**: SciGraph AI (`data/`)
- **Date Ingested**: October 2026
- **Source**: OpenAlex REST API (`https://api.openalex.org/works`)
- **Scope**: 16,500 unique peer-reviewed scientific publications with full abstracts across 5 major scientific disciplines spanning publication years 2010 through 2020.
- **Design Objective**: Evaluate 5-year citation trajectory and cohort-relative impact tier using strictly historical information observed as of $T_{\text{cutoff}} = \text{pub\_year} + 2$.

---

## 2. Disciplinary & Temporal Stratification

The corpus is balanced across 5 diverse fields (defined by OpenAlex Level-0/1 topic taxonomy) and 11 publication cohorts (2010–2020):

| Domain / Field | OpenAlex Field ID | Ingested Works | Publication Years |
| :--- | :--- | :--- | :--- |
| **Computer Science** | `fields/17` | 3,300 works (300/cohort) | 2010 – 2020 |
| **Medicine** | `fields/27` | 3,300 works (300/cohort) | 2010 – 2020 |
| **Physics and Astronomy** | `fields/31` | 3,300 works (300/cohort) | 2010 – 2020 |
| **Biochemistry, Genetics & Molecular Biology** | `fields/13` | 3,300 works (300/cohort) | 2010 – 2020 |
| **Engineering** | `fields/22` | 3,300 works (300/cohort) | 2010 – 2020 |
| **Total Corpus** | — | **16,500 unique works** | **2010 – 2020** |

### Verified Table Sizes:
- `works.parquet`: 16,500 rows (2.1 MB)
- `authors.parquet`: 104,906 rows (3.4 MB)
- `institutions.parquet`: 7,151 rows (227.3 KB)
- `authorships.parquet`: 167,202 rows (2.0 MB)
- `sources.parquet` (Venues): 2,441 rows (104.5 KB)
- `topics.parquet`: 1,196 rows (43.2 KB)
- `citation_edges.parquet`: 1,760,775 rows (16.9 MB)
- `counts_by_year.parquet`: 200,490 rows (540.4 KB)
- `processed/train.parquet`: 10,500 rows ($\le 2016$)
- `processed/val.parquet`: 1,500 rows ($= 2017$)
- `processed/test.parquet`: 4,500 rows ($\ge 2018$)

---

## 3. Relational Schema Architecture

The dataset is stored in normalized Apache Parquet tables under `data/`:

### `works.parquet`
Primary entity representing scientific publications.
- `work_id` (string, PK): OpenAlex URI (e.g., `https://openalex.org/W...`).
- `title` (string): Full publication title.
- `doi` (string, optional): Digital Object Identifier URI.
- `publication_year` (int): Year of formal publication ($Y \in [2010, 2020]$).
- `publication_date` (string, optional): ISO publication date.
- `is_oa` (bool): Open access flag.
- `oa_status` (string): Access status classification (`gold`, `green`, `hybrid`, `bronze`, `closed`).
- `oa_url` (string, optional): Canonical full-text access URL.
- `apc_list_usd` (float, optional): Listed Article Processing Charge in USD.
- `apc_paid_usd` (float, optional): Documented APC paid in USD.
- `fwci` (float, optional): Field-Weighted Citation Impact from OpenAlex.
- `openalex_percentile` (float, optional): OpenAlex citation-normalized percentile.
- `primary_topic_id` (string, FK): ID of primary research topic.
- `primary_field_id` (string): ID of parent scientific field.
- `primary_field_name` (string): Name of parent scientific field.
- `primary_subfield_id` (string): ID of parent subfield.
- `primary_subfield_name` (string): Name of parent subfield.
- `venue_id` (string, FK): OpenAlex Source / Venue ID.
- `venue_name` (string): Journal or conference name.
- `venue_issn_l` (string, optional): Linking ISSN for external scientometric matching (SCImago/CORE).
- `venue_type` (string): Type of venue (`journal`, `conference`, `repository`, etc.).
- `venue_publisher` (string, optional): Publishing organization.
- `num_authors` (int): Exact count of authors listed on the paper.
- `referenced_works_count` (int): Number of outgoing citation references.
- `cutoff_year` (int): Scientometric cutoff year ($T_{\text{cutoff}} = Y + 2$).
- `citations_at_cutoff` (int): Sum of annual citations received strictly on or before $T_{\text{cutoff}}$.
- `future_citations_5y` (int): Sum of annual citations received strictly between $T_{\text{cutoff}} < t \le Y + 5$.
- `cohort_percentile` (float): Percentile rank of `future_citations_5y` within $(F_i, Y_i)$ stratum.
- `impact_class` (int): Target label:
  - `0`: Low Impact ($< 50\text{th}$ percentile)
  - `1`: Medium Impact ($50\text{th} - 90\text{th}$ percentile)
  - `2`: High Impact ($\ge 90\text{th}$ percentile)
- `temporal_split` (string): Split partition (`train`, `val`, `test`).

### `authors.parquet`
- `author_id` (string, PK): OpenAlex Author ID.
- `display_name` (string): Full name of researcher.
- `orcid` (string, optional): Verified ORCID profile URI.

### `institutions.parquet`
- `institution_id` (string, PK): OpenAlex Institution ID.
- `display_name` (string): Organization name.
- `country_code` (string, optional): ISO two-letter country code.
- `type` (string, optional): Entity classification (`education`, `company`, `government`, `facility`, etc.).

### `authorships.parquet`
- `work_id` (string, FK): Publication ID.
- `author_id` (string, FK): Author ID.
- `institution_id` (string, FK, optional): Affiliated institution ID.
- `author_position` (string): Position tier (`first`, `middle`, `last`).

### `sources.parquet` (Venues)
- `venue_id` (string, PK): OpenAlex Source ID.
- `display_name` (string): Official venue title.
- `issn_l` (string, optional): Linking ISSN.
- `type` (string): Publication medium.
- `host_organization_name` (string, optional): Academic publisher.
- `is_in_doaj` (bool): Directory of Open Access Journals inclusion.

### `topics.parquet`
- `topic_id` (string, PK): Topic ID.
- `topic_name` (string): Granular topic descriptor.
- `subfield_id` (string): Subfield identifier.
- `subfield_name` (string): Subfield name.
- `field_id` (string): Discipline field identifier.
- `field_name` (string): Discipline field name.
- `domain_id` (string): Broad scientific domain identifier.
- `domain_name` (string): Broad scientific domain name.

### `citation_edges.parquet`
- `citing_work_id` (string, FK): OpenAlex work initiating the citation.
- `referenced_work_id` (string): OpenAlex ID of referenced paper.

### `counts_by_year.parquet`
- `work_id` (string, FK): Publication ID.
- `year` (int): Calendar observation year.
- `cited_by_count` (int): Exact citations accrued during that calendar year.

---

## 4. Anti-Leakage Protocol (Rule R3)

To ensure zero time leakage:
1. **Historical Features**: All paper features (citations, venue impact, author metrics) used for model training and inference must be computed strictly as of $T_{\text{cutoff}} = \text{pub\_year} + 2$.
2. **Evaluation Window**: The target citation accumulation is measured across the remaining 3 years of the 5-year evaluation window:
   $$C_{\text{target}} = \sum_{t = \text{pub\_year} + 3}^{\text{pub\_year} + 5} \text{cited\_by\_count}_t$$
3. **Temporal Partitioning**:
   - **Training Set**: Publications from years $2010 \le Y \le 2016$.
   - **Validation Set**: Publications from year $Y = 2017$.
   - **Test Set**: Publications from years $2018 \le Y \le 2020$.
   Evaluating on strictly future cohorts prevents temporal distribution shift leakage.

---

## 5. Cohort-Relative Label Formulation

Direct citation counts cannot be compared across disciplines or historical eras without introducing severe bias (e.g., biomedical papers accrue citations at 3–5x the rate of pure mathematics papers).
- Each paper $i$ is assigned a percentile rank $P_i$ within its specific field-year cohort $(F_i, Y_i)$:
  $$P_i = \text{Rank}\left(C_{\text{target}, i}\right) / N_{(F_i, Y_i)} \times 100\%$$
- **Class Boundaries**:
  - Class 0 (Low): $P_i < 50.0\%$
  - Class 1 (Medium): $50.0\% \le P_i < 90.0\%$
  - Class 2 (High Impact): $P_i \ge 90.0\%$

---

## 6. Sampling Limitations & Known Biases

- **Abstract Filtering**: Requires `has_abstract:true`, excluding short editorials, errata, and works without indexed abstracts.
- **APC Coverage**: APC list/paid amounts are available primarily for open access and hybrid journals with public pricing. Closed subscription journals report `null`.
- **Open Access Selection Effect**: In scientometrics literature, open access papers often correlate with higher citations, but this is partly driven by author self-selection (authors choose to publish their best work as OA). This dataset reports empirical association, not causal claims.
