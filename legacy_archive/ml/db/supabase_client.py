"""Supabase Database Client & Upsert Layer."""

import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class SupabaseDBClient:
    """Supabase Postgres interface for storing entities & edge relations."""

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        self._client = None

        if self.url and self.key and not self.url.startswith("https://your-project"):
            try:
                from supabase import create_client, Client
                self._client: Optional[Client] = create_client(self.url, self.key)
            except Exception as e:
                print(f"[SupabaseDBClient] Warning: Could not connect to Supabase ({e}). Operating in local offline mode.")
        else:
            print("[SupabaseDBClient] Info: No live Supabase credentials configured. Operating in local offline mode.")

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    def upsert_papers(self, papers: List[Dict[str, Any]]) -> bool:
        """Upsert paper records into Supabase 'papers' table."""
        if not self._client or not papers:
            return False
        try:
            self._client.table("papers").upsert(papers).execute()
            return True
        except Exception as e:
            print(f"[SupabaseDBClient] Error upserting papers: {e}")
            return False

    def upsert_authors(self, authors: List[Dict[str, Any]]) -> bool:
        """Upsert author records into Supabase 'authors' table."""
        if not self._client or not authors:
            return False
        try:
            self._client.table("authors").upsert(authors).execute()
            return True
        except Exception as e:
            print(f"[SupabaseDBClient] Error upserting authors: {e}")
            return False

    def upsert_institutions(self, institutions: List[Dict[str, Any]]) -> bool:
        """Upsert institution records into Supabase 'institutions' table."""
        if not self._client or not institutions:
            return False
        try:
            self._client.table("institutions").upsert(institutions).execute()
            return True
        except Exception as e:
            print(f"[SupabaseDBClient] Error upserting institutions: {e}")
            return False

    def upsert_topics(self, topics: List[Dict[str, Any]]) -> bool:
        """Upsert topic records into Supabase 'topics' table."""
        if not self._client or not topics:
            return False
        try:
            self._client.table("topics").upsert(topics).execute()
            return True
        except Exception as e:
            print(f"[SupabaseDBClient] Error upserting topics: {e}")
            return False

    def upsert_authorships(self, authorships: List[Dict[str, Any]]) -> bool:
        """Upsert authorship edge records into Supabase 'authorships' table."""
        if not self._client or not authorships:
            return False
        try:
            self._client.table("authorships").upsert(authorships).execute()
            return True
        except Exception as e:
            print(f"[SupabaseDBClient] Error upserting authorships: {e}")
            return False

    def upsert_paper_citations(self, citations: List[Dict[str, Any]]) -> bool:
        """Upsert paper citation edge records into Supabase 'paper_citations' table."""
        if not self._client or not citations:
            return False
        try:
            self._client.table("paper_citations").upsert(citations).execute()
            return True
        except Exception as e:
            print(f"[SupabaseDBClient] Error upserting paper citations: {e}")
            return False
