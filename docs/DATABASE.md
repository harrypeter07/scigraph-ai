# Database Architecture & Supabase Schema

> **Storage Strategy**: Dual-Tier Architecture  
> 1. **Supabase (Postgres + pgvector)**: Central queryable source of truth.  
> 2. **Local Parquet Cache (`data/interim/`, `data/processed/`)**: High-performance local columnar storage for fast batch training iterations.  

---

## 1. Supabase Entity-Relationship Diagram (ASCII)

```
 +------------------+           +--------------------+           +---------------------+
 |      PAPERS      |           |     AUTHORSHIPS    |           |       AUTHORS       |
 +------------------+           +--------------------+           +---------------------+
 | id (PK)          |<---------1| paper_id (FK)      |           | id (PK)             |
 | title            |           | author_id (FK)    |---------->| display_name        |
 | publication_year |           | institution_id(FK) |           | works_count         |
 | publication_date |           +--------------------+           +---------------------+
 | abstract_text    |                                                       |
 | primary_topic_id |                                                       |
 +------------------+                                                       v
         |                                                       +---------------------+
         | 1                                                     |     AFFILIATIONS    |
         v                                                       +---------------------+
 +------------------+                                            | author_id (FK)      |
 |  PAPER_TOPICS    |                                            | institution_id (FK) |
 +------------------+                                            +---------------------+
 | paper_id (FK)    |                                                       |
 | topic_id (FK)    |                                                       v
 +------------------+                                            +---------------------+
         |                                                       |     INSTITUTIONS    |
         v                                                       +---------------------+
 +------------------+                                            | id (PK)             |
 |      TOPICS      |                                            | display_name        |
 +------------------+                                            | country_code        |
 | id (PK)          |                                            +---------------------+
 | display_name     |
 +------------------+
```

---

## 2. Supabase DDL Schema

```sql
-- Enable pgvector extension for abstract/title embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Papers Table
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    doi TEXT,
    title TEXT NOT NULL,
    publication_year INT NOT NULL,
    publication_date DATE,
    abstract_text TEXT,
    primary_topic_id TEXT,
    referenced_works_count INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Paper Citations Edge Table
CREATE TABLE IF NOT EXISTS paper_citations (
    citing_paper_id TEXT REFERENCES papers(id),
    cited_paper_id TEXT REFERENCES papers(id),
    citation_year INT,
    PRIMARY KEY (citing_paper_id, cited_paper_id)
);

-- Authors Table
CREATE TABLE IF NOT EXISTS authors (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Institutions Table
CREATE TABLE IF NOT EXISTS institutions (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    country_code VARCHAR(10),
    type TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Topics Table
CREATE TABLE IF NOT EXISTS topics (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    level INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Authorships Edge Table
CREATE TABLE IF NOT EXISTS authorships (
    paper_id TEXT REFERENCES papers(id),
    author_id TEXT REFERENCES authors(id),
    institution_id TEXT REFERENCES institutions(id),
    author_position INT,
    PRIMARY KEY (paper_id, author_id)
);

-- Paper Topics Edge Table
CREATE TABLE IF NOT EXISTS paper_topics (
    paper_id TEXT REFERENCES papers(id),
    topic_id TEXT REFERENCES topics(id),
    score FLOAT,
    PRIMARY KEY (paper_id, topic_id)
);

-- Paper Embeddings Table (Vector Search support)
CREATE TABLE IF NOT EXISTS paper_embeddings (
    paper_id TEXT PRIMARY KEY REFERENCES papers(id),
    embedding vector(384), -- e.g. sentence-transformers/all-MiniLM-L6-v2
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

---

## 3. Parquet Local Fast-Path

- **Script**: `ml/db/parquet_exporter.py`
- **Output Tables**: `data/interim/papers.parquet`, `data/interim/edges.parquet`, `data/processed/features_l2.parquet`.
- **Reproducibility Guarantee**: Parquet datasets can be regenerated deterministically at any time from raw JSONL archives or Supabase query dumps.
