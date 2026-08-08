"""Preprocessing & Data Cleaning Pipeline.

Parses raw OpenAlex JSONL dumps, deduplicates entities, extracts relationships,
and builds clean interim Parquet tables in data/interim/.
"""

import os
import json
import glob
import pandas as pd
from typing import List, Dict, Any, Tuple

from ml.acquisition.openalex import parse_openalex_record


class OpenAlexPreprocessor:
    """Processor to convert raw JSONL records to structured interim Parquet datasets."""

    def __init__(self, raw_dir: str = "data/raw/openalex", interim_dir: str = "data/interim"):
        self.raw_dir = raw_dir
        self.interim_dir = interim_dir
        os.makedirs(self.interim_dir, exist_ok=True)

    def load_raw_jsonl(self) -> List[Dict[str, Any]]:
        """Load all raw OpenAlex work records from raw JSONL files."""
        pattern = os.path.join(self.raw_dir, "*.jsonl")
        jsonl_files = glob.glob(pattern)
        records = []

        for filepath in jsonl_files:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

        return records

    def process_records(self, raw_records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Parse and deduplicate raw records into relational DataFrames."""
        papers_list = []
        authors_dict: Dict[str, Dict[str, Any]] = {}
        institutions_dict: Dict[str, Dict[str, Any]] = {}
        authorships_list = []
        citations_list = []

        seen_paper_ids = set()

        for raw_work in raw_records:
            parsed = parse_openalex_record(raw_work)
            paper = parsed["paper"]
            paper_id = paper["id"]

            if not paper_id or paper_id in seen_paper_ids:
                continue

            seen_paper_ids.add(paper_id)
            papers_list.append(paper)

            # Authors & Institutions
            for auth in parsed["authors"]:
                authors_dict[auth["id"]] = auth

            for inst in parsed["institutions"]:
                institutions_dict[inst["id"]] = inst

            for authorship in parsed["authorships"]:
                authorships_list.append(authorship)

            # Citations (referenced_works)
            for ref_id in paper.get("referenced_works", []):
                citations_list.append({
                    "citing_paper_id": paper_id,
                    "cited_paper_id": ref_id,
                    "citation_year": paper["publication_year"]
                })

        papers_df = pd.DataFrame(papers_list)
        authors_df = pd.DataFrame(list(authors_dict.values())) if authors_dict else pd.DataFrame(columns=["id", "display_name"])
        institutions_df = pd.DataFrame(list(institutions_dict.values())) if institutions_dict else pd.DataFrame(columns=["id", "display_name", "country_code", "type"])
        authorships_df = pd.DataFrame(authorships_list) if authorships_list else pd.DataFrame(columns=["paper_id", "author_id", "institution_id", "author_position"])
        citations_df = pd.DataFrame(citations_list) if citations_list else pd.DataFrame(columns=["citing_paper_id", "cited_paper_id", "citation_year"])

        return papers_df, authors_df, institutions_df, authorships_df, citations_df

    def save_interim_parquet(self) -> Dict[str, str]:
        """Run preprocessing pipeline and export interim Parquet datasets."""
        raw_records = self.load_raw_jsonl()
        if not raw_records:
            raise ValueError(f"No raw JSONL records found in {self.raw_dir}")

        papers_df, authors_df, institutions_df, authorships_df, citations_df = self.process_records(raw_records)

        output_paths = {
            "papers": os.path.join(self.interim_dir, "papers.parquet"),
            "authors": os.path.join(self.interim_dir, "authors.parquet"),
            "institutions": os.path.join(self.interim_dir, "institutions.parquet"),
            "authorships": os.path.join(self.interim_dir, "authorships.parquet"),
            "citations": os.path.join(self.interim_dir, "paper_citations.parquet"),
        }

        # Convert list/dict columns to json string format for Parquet compatibility if needed
        papers_df_copy = papers_df.copy()
        if "counts_by_year" in papers_df_copy.columns:
            papers_df_copy["counts_by_year"] = papers_df_copy["counts_by_year"].apply(json.dumps)
        if "referenced_works" in papers_df_copy.columns:
            papers_df_copy["referenced_works"] = papers_df_copy["referenced_works"].apply(json.dumps)

        papers_df_copy.to_parquet(output_paths["papers"], index=False)
        authors_df.to_parquet(output_paths["authors"], index=False)
        institutions_df.to_parquet(output_paths["institutions"], index=False)
        authorships_df.to_parquet(output_paths["authorships"], index=False)
        citations_df.to_parquet(output_paths["citations"], index=False)

        return output_paths


if __name__ == "__main__":
    preprocessor = OpenAlexPreprocessor()
    paths = preprocessor.save_interim_parquet()
    print("Interim Parquet datasets generated successfully:", paths)
