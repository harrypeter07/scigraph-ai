# Data Dictionary: SciGraph AI

This dictionary defines all entity fields, data types, system roles, leakage risk assessments, and MVP status for the SciGraph AI platform.

---

## 1. Paper Entity Schema

| Field Name | Source | Data Type | System Role | Leakage Risk | MVP Status | Description |
|---|---|---|---|---|---|---|
| `id` | OpenAlex | String (URI/ID) | Identifier | NONE | Yes | Unique OpenAlex paper ID (e.g. `https://openalex.org/W2741809807`) |
| `title` | OpenAlex | String | Feature / Text | NONE | Yes | Title of the publication |
| `publication_year` | OpenAlex | Integer | Snapshot / Split | NONE | Yes | Publication calendar year ($Y_{pub}$) |
| `publication_date` | OpenAlex | String (ISO) | Temporal Filter | NONE | Yes | Exact ISO publication date |
| `abstract_inverted_index` | OpenAlex | Dict / JSON | Text Feature | NONE | Yes | Reconstructible abstract text |
| `cited_by_count` | OpenAlex | Integer | Raw Material | **HIGH** | No (Raw) | Current lifetime citation count (MUST NOT be used directly as feature) |
| `counts_by_year` | OpenAlex | List[Dict] | Target Material | **MEDIUM** | Yes | Annual citation counts breakdown used to compute historical trajectory & labels |
| `referenced_works` | OpenAlex | List[String] | Graph Edge | **LOW** | Yes | Outgoing citation links to antecedent papers |
| `referenced_works_count` | OpenAlex | Integer | Feature | NONE | Yes | Number of references cited by paper |
| `primary_topic` | OpenAlex | Dict / String | Cohort / Graph | NONE | Yes | Primary OpenAlex research subfield classification |
| `concepts` | OpenAlex | List[Dict] | Graph Node | NONE | Yes | Concept tags with confidence scores |

---

## 2. Author Entity Schema

| Field Name | Source | Data Type | System Role | Leakage Risk | MVP Status | Description |
|---|---|---|---|---|---|---|
| `id` | OpenAlex | String | Identifier | NONE | Yes | Unique author OpenAlex ID |
| `display_name` | OpenAlex | String | Metadata | NONE | Yes | Author's full name |
| `works_count` | OpenAlex | Integer | Feature | **HIGH** | No | Aggregate lifetime works count (Must snapshot historically if used) |
| `cited_by_count` | OpenAlex | Integer | Feature | **HIGH** | No | Lifetime citations (Must snapshot historically if used) |
| `last_known_institution` | OpenAlex | String | Graph Node | **MEDIUM** | Yes | Institution affiliation ID |

---

## 3. Institution Entity Schema

| Field Name | Source | Data Type | System Role | Leakage Risk | MVP Status | Description |
|---|---|---|---|---|---|---|
| `id` | OpenAlex | String | Identifier | NONE | Yes | OpenAlex institution ID |
| `display_name` | OpenAlex | String | Metadata | NONE | Yes | Institution name |
| `country_code` | OpenAlex | String | Categorical Feat | NONE | Yes | ISO country code |
| `type` | OpenAlex | String | Categorical Feat | NONE | Yes | Type (Education, Facility, Company, etc.) |

---

## 4. Topic / Concept Entity Schema

| Field Name | Source | Data Type | System Role | Leakage Risk | MVP Status | Description |
|---|---|---|---|---|---|---|
| `id` | OpenAlex | String | Identifier | NONE | Yes | Concept / Topic ID |
| `display_name` | OpenAlex | String | Metadata | NONE | Yes | Human readable topic label |
| `level` | OpenAlex | Integer | Taxonomy Tier | NONE | Yes | Hierarchy level (0 = field, 1 = subfield, etc.) |

---

## 5. Target & Label Schema

| Field Name | Source | Data Type | System Role | Leakage Risk | MVP Status | Description |
|---|---|---|---|---|---|---|
| `delta_citations_5y` | Derived | Integer | Target Raw | N/A (Ground Truth) | Yes | Citations accumulated in $Y_{pub}+1 \dots Y_{pub}+5$ |
| `impact_cohort_percentile` | Derived | Float | Target Percentile | N/A (Ground Truth) | Yes | Percentile rank within $(Y_{pub}, \text{Topic})$ cohort |
| `impact_label` | Derived | Integer (0, 1, 2) | Class Target | N/A (Ground Truth) | Yes | 0: Low (<50%), 1: Medium (50-90%), 2: High (>=90%) |
