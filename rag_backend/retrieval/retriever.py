"""
Query-side counterpart to the vector store.

Retrieval pipeline:

    User query
        ↓
    Query embedding
        ↓
    Broad Pinecone semantic search
        ↓
    Generic lexical / phrase reranking
        ↓
    Page expansion
        ↓
    Complete evidence
        ↓
    Reflection Agent

Important:
    This file contains NO document-specific rules.

It does not know about:
    - marksheets
    - Deepak
    - semesters
    - invoices
    - resumes
    - certificates

Instead, it extracts useful terms and concepts from whatever
the user asks.
"""

import re

from config import settings
from core.models import RetrievalResult
from embedding.embedder import Embedder
from storage.vector_store import VectorStoreRepository


class Retriever:

    # ------------------------------------------------------------------
    # Generic stop words
    # ------------------------------------------------------------------

    STOP_WORDS = {
        "the",
        "what",
        "which",
        "how",
        "many",
        "much",
        "did",
        "does",
        "do",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "can",
        "could",
        "would",
        "should",
        "of",
        "in",
        "on",
        "at",
        "for",
        "to",
        "from",
        "with",
        "and",
        "or",
        "a",
        "an",
        "his",
        "her",
        "their",
        "its",
        "this",
        "that",
        "these",
        "those",
        "give",
        "tell",
        "me",
        "please",
        "show",
        "find",
        "get",
        "provide",
    }

    # ------------------------------------------------------------------
    # Generic ordinal normalization
    #
    # This is NOT marksheet-specific.
    #
    # Examples:
    #
    #   1st      -> first
    #   2nd      -> second
    #   3rd      -> third
    #   4th      -> fourth
    # ------------------------------------------------------------------

    ORDINAL_MAP = {
        "1": "first",
        "1st": "first",
        "first": "first",

        "2": "second",
        "2nd": "second",
        "second": "second",

        "3": "third",
        "3rd": "third",
        "third": "third",

        "4": "fourth",
        "4th": "fourth",
        "fourth": "fourth",

        "5": "fifth",
        "5th": "fifth",
        "fifth": "fifth",

        "6": "sixth",
        "6th": "sixth",
        "sixth": "sixth",

        "7": "seventh",
        "7th": "seventh",
        "seventh": "seventh",

        "8": "eighth",
        "8th": "eighth",
        "eighth": "eighth",

        "9": "ninth",
        "9th": "ninth",
        "ninth": "ninth",

        "10": "tenth",
        "10th": "tenth",
        "tenth": "tenth",
    }

    # ------------------------------------------------------------------
    # Generic abbreviations
    #
    # These allow different representations of the same concept.
    # ------------------------------------------------------------------

    NORMALIZATION_MAP = {
        "sem": "semester",
        "sem.": "semester",

        "yr": "year",
        "yr.": "year",

        "no": "number",
        "no.": "number",

        "amt": "amount",
        "amt.": "amount",

        "qty": "quantity",
        "qty.": "quantity",
    }

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStoreRepository,
        score_threshold: float | None = None,
    ):

        self.embedder = embedder

        self.vector_store = vector_store

        self.score_threshold = (
            score_threshold
            if score_threshold is not None
            else settings.score_threshold
        )

        # --------------------------------------------------------------
        # Retrieve more candidates than the final top_k.
        #
        # Example:
        #
        # settings.top_k = 6
        #
        # candidate_k = 18
        #
        # This gives reranking enough material to find the correct page.
        # --------------------------------------------------------------

        self.candidate_k = max(
            settings.top_k * 3,
            18,
        )

    # ==================================================================
    # MAIN RETRIEVAL
    # ==================================================================

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:

        top_k = top_k or settings.top_k

        # --------------------------------------------------------------
        # 1. Create query embedding
        # --------------------------------------------------------------

        query_vector = (
            self.embedder.embed_text(
                query
            )
        )

        # --------------------------------------------------------------
        # 2. Broad semantic retrieval
        # --------------------------------------------------------------

        candidates = (
            self.vector_store.query(
                query_vector,
                top_k=self.candidate_k,
            )
        )

        if not candidates:

            print(
                "\n[RAG DEBUG] "
                "No candidates retrieved."
            )

            return []

        # --------------------------------------------------------------
        # 3. Extract query information
        # --------------------------------------------------------------

        query_terms = (
            self._extract_query_terms(
                query
            )
        )

        query_phrases = (
            self._extract_important_phrases(
                query
            )
        )

        normalized_query = (
            self._normalize_text(
                query
            )
        )

        # --------------------------------------------------------------
        # 4. Debug
        # --------------------------------------------------------------

        print(
            "\n"
            + "=" * 100
        )

        print(
            "[RAG DEBUG] USER QUERY"
        )

        print(
            query
        )

        print(
            "\n[RAG DEBUG] QUERY TERMS"
        )

        print(
            query_terms
        )

        print(
            "\n[RAG DEBUG] IMPORTANT PHRASES"
        )

        print(
            query_phrases
        )

        print(
            "\n[RAG DEBUG] NORMALIZED QUERY"
        )

        print(
            normalized_query
        )

        print(
            "\n[RAG DEBUG] CANDIDATES"
        )

        print(
            f"Candidate count: "
            f"{len(candidates)}"
        )

        # --------------------------------------------------------------
        # 5. Rerank candidates
        # --------------------------------------------------------------

        scored_candidates = []

        for result in candidates:

            text = (
                result.chunk.text
                or ""
            )

            normalized_text = (
                self._normalize_text(
                    text
                )
            )

            lexical_score = (
                self._lexical_score(
                    query_terms=query_terms,
                    text=normalized_text,
                )
            )

            phrase_score = (
                self._phrase_score(
                    query_phrases=query_phrases,
                    text=normalized_text,
                )
            )

            specificity_score = (
                self._specificity_score(
                    query=query,
                    text=text,
                )
            )

            # ----------------------------------------------------------
            # Semantic similarity remains dominant.
            #
            # Lexical and phrase scores only help distinguish documents
            # that are semantically very similar.
            # ----------------------------------------------------------

            final_score = (
                float(result.score)
                + lexical_score
                + phrase_score
                + specificity_score
            )

            scored_candidates.append(
                (
                    final_score,
                    result,
                    lexical_score,
                    phrase_score,
                    specificity_score,
                )
            )

        # --------------------------------------------------------------
        # Highest score first
        # --------------------------------------------------------------

        scored_candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        ranked_results = [
            item[1]
            for item in scored_candidates[
                :top_k
            ]
        ]

        # --------------------------------------------------------------
        # 6. Debug reranking
        # --------------------------------------------------------------

        print(
            "\n[RAG DEBUG] RERANKED RESULTS"
        )

        for i, (
            final_score,
            result,
            lexical_score,
            phrase_score,
            specificity_score,
        ) in enumerate(
            scored_candidates[:top_k],
            1,
        ):

            print(
                "\n"
                + "-" * 80
            )

            print(
                f"RESULT #{i}"
            )

            print(
                f"Original semantic score: "
                f"{result.score:.6f}"
            )

            print(
                f"Lexical score: "
                f"{lexical_score:.6f}"
            )

            print(
                f"Phrase score: "
                f"{phrase_score:.6f}"
            )

            print(
                f"Specificity score: "
                f"{specificity_score:.6f}"
            )

            print(
                f"FINAL SCORE: "
                f"{final_score:.6f}"
            )

            print(
                f"Source: "
                f"{result.source_filename}"
            )

            print(
                f"Page: "
                f"{result.chunk.page_number}"
            )

            print(
                f"Chunk index: "
                f"{result.chunk.chunk_index}"
            )

            print(
                f"Characters: "
                f"{len(result.chunk.text)}"
            )

            print(
                "\nTEXT:"
            )

            print(
                result.chunk.text[:1200]
            )

        # --------------------------------------------------------------
        # 7. Expand relevant pages
        # --------------------------------------------------------------
        #
        # If a relevant chunk is found on a page, retrieve all chunks
        # belonging to that page.
        #
        # This is generic.
        # --------------------------------------------------------------

        expanded_results = (
            self.vector_store.expand_results_by_page(
                results=ranked_results,
                query_vector=query_vector,
                max_pages=3,
            )
        )

        # --------------------------------------------------------------
        # 8. Debug page expansion
        # --------------------------------------------------------------

        print(
            "\n[RAG DEBUG] PAGE EXPANSION"
        )

        print(
            f"Before expansion: "
            f"{len(ranked_results)} chunks"
        )

        print(
            f"After expansion: "
            f"{len(expanded_results)} chunks"
        )

        for result in expanded_results:

            print(
                "\n"
                + "-" * 80
            )

            print(
                f"Source: "
                f"{result.source_filename}"
            )

            print(
                f"Page: "
                f"{result.chunk.page_number}"
            )

            print(
                f"Chunk index: "
                f"{result.chunk.chunk_index}"
            )

            print(
                f"Characters: "
                f"{len(result.chunk.text)}"
            )

            print(
                "\nTEXT:"
            )

            print(
                result.chunk.text[:1200]
            )

        print(
            "\n"
            + "=" * 100
        )

        return expanded_results

    # ==================================================================
    # QUERY TERM EXTRACTION
    # ==================================================================

    def _extract_query_terms(
        self,
        query: str,
    ) -> list[str]:

        words = re.findall(
            r"[a-zA-Z0-9]+",
            query.lower(),
        )

        terms = []

        for word in words:

            # ----------------------------------------------------------
            # Ignore normal stop words
            # ----------------------------------------------------------

            if word in self.STOP_WORDS:
                continue

            # ----------------------------------------------------------
            # Keep numbers because they can be important.
            #
            # Example:
            #
            # 2024
            # 10
            # 500
            # ----------------------------------------------------------

            if len(word) < 2:
                continue

            # ----------------------------------------------------------
            # Normalize abbreviations
            # ----------------------------------------------------------

            normalized = (
                self.NORMALIZATION_MAP.get(
                    word,
                    word,
                )
            )

            # ----------------------------------------------------------
            # Normalize ordinals
            # ----------------------------------------------------------

            normalized = (
                self.ORDINAL_MAP.get(
                    normalized,
                    normalized,
                )
            )

            terms.append(
                normalized
            )

        # --------------------------------------------------------------
        # Remove duplicates while preserving order.
        # --------------------------------------------------------------

        return list(
            dict.fromkeys(
                terms
            )
        )

    # ==================================================================
    # IMPORTANT PHRASES
    # ==================================================================

    def _extract_important_phrases(
        self,
        query: str,
    ) -> list[str]:

        normalized = (
            self._normalize_text(
                query
            )
        )

        phrases = []

        # --------------------------------------------------------------
        # Detect ordinal + common structural noun.
        #
        # Examples:
        #
        # first semester
        # second year
        # third quarter
        # fourth section
        # --------------------------------------------------------------

        ordinal_pattern = (
            r"\b(first|second|third|fourth|"
            r"fifth|sixth|seventh|eighth|"
            r"ninth|tenth)\s+"
            r"(semester|year|quarter|"
            r"section|chapter|part|stage|"
            r"phase)\b"
        )

        matches = re.findall(
            ordinal_pattern,
            normalized,
        )

        for ordinal, noun in matches:

            phrases.append(
                f"{ordinal} {noun}"
            )

        # --------------------------------------------------------------
        # Detect "semester 1", "year 2", etc.
        # --------------------------------------------------------------

        number_pattern = (
            r"\b(semester|year|quarter|"
            r"section|chapter|part|stage|phase)"
            r"\s+(\d+)\b"
        )

        matches = re.findall(
            number_pattern,
            normalized,
        )

        for noun, number in matches:

            ordinal = (
                self.ORDINAL_MAP.get(
                    number,
                    number,
                )
            )

            phrases.append(
                f"{ordinal} {noun}"
            )

        # --------------------------------------------------------------
        # Detect hyphenated abbreviations:
        #
        # sem-i
        # sem-1
        # section-2
        # chapter-3
        # --------------------------------------------------------------

        abbreviation_pattern = (
            r"\b(sem|semester|year|"
            r"section|chapter|part)"
            r"[-\s]"
            r"([ivx]+|\d+)\b"
        )

        matches = re.findall(
            abbreviation_pattern,
            normalized,
        )

        for noun, value in matches:

            value = value.lower()

            roman_map = {
                "i": "first",
                "ii": "second",
                "iii": "third",
                "iv": "fourth",
                "v": "fifth",
                "vi": "sixth",
                "vii": "seventh",
                "viii": "eighth",
                "ix": "ninth",
                "x": "tenth",
            }

            ordinal = roman_map.get(
                value,
                self.ORDINAL_MAP.get(
                    value,
                    value,
                ),
            )

            normalized_noun = (
                self.NORMALIZATION_MAP.get(
                    noun,
                    noun,
                )
            )

            phrases.append(
                f"{ordinal} "
                f"{normalized_noun}"
            )

        # --------------------------------------------------------------
        # Important quoted phrases
        # --------------------------------------------------------------

        quoted = re.findall(
            r'"([^"]+)"',
            query,
        )

        for phrase in quoted:

            phrase = (
                self._normalize_text(
                    phrase
                )
            )

            if phrase:
                phrases.append(
                    phrase
                )

        # --------------------------------------------------------------
        # Remove duplicates
        # --------------------------------------------------------------

        return list(
            dict.fromkeys(
                phrases
            )
        )

    # ==================================================================
    # NORMALIZE TEXT
    # ==================================================================

    def _normalize_text(
        self,
        text: str,
    ) -> str:

        text = text.lower()

        # --------------------------------------------------------------
        # Replace punctuation with spaces.
        # --------------------------------------------------------------

        text = re.sub(
            r"[^a-z0-9\s\-]",
            " ",
            text,
        )

        # --------------------------------------------------------------
        # Normalize common abbreviations.
        # --------------------------------------------------------------

        for short, full in (
            self.NORMALIZATION_MAP.items()
        ):

            text = re.sub(
                rf"\b{re.escape(short)}\b",
                full,
                text,
            )

        # --------------------------------------------------------------
        # Normalize ordinal representations.
        # --------------------------------------------------------------

        for short, full in (
            sorted(
                self.ORDINAL_MAP.items(),
                key=lambda x: len(x[0]),
                reverse=True,
            )
        ):

            text = re.sub(
                rf"\b{re.escape(short)}\b",
                full,
                text,
            )

        # --------------------------------------------------------------
        # Collapse whitespace.
        # --------------------------------------------------------------

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ==================================================================
    # LEXICAL SCORE
    # ==================================================================

    def _lexical_score(
        self,
        query_terms: list[str],
        text: str,
    ) -> float:

        if not query_terms:
            return 0.0

        if not text:
            return 0.0

        matched = 0

        for term in query_terms:

            # Word-boundary matching avoids accidental partial matches.
            pattern = (
                rf"\b{re.escape(term)}\b"
            )

            if re.search(
                pattern,
                text,
            ):

                matched += 1

        if matched == 0:
            return 0.0

        ratio = (
            matched
            / len(query_terms)
        )

        # Small generic boost.
        return 0.08 * ratio

    # ==================================================================
    # PHRASE SCORE
    # ==================================================================

    def _phrase_score(
        self,
        query_phrases: list[str],
        text: str,
    ) -> float:

        if not query_phrases:
            return 0.0

        if not text:
            return 0.0

        matched = 0

        for phrase in query_phrases:

            if phrase in text:
                matched += 1

        if matched == 0:
            return 0.0

        ratio = (
            matched
            / len(query_phrases)
        )

        # Stronger than individual-word matching.
        #
        # This allows:
        #
        # "first semester"
        #
        # to outrank a page that only contains:
        #
        # "semester"
        #
        # but not "first".
        return 0.20 * ratio

    # ==================================================================
    # SPECIFICITY SCORE
    # ==================================================================

    def _specificity_score(
        self,
        query: str,
        text: str,
    ) -> float:

        query_normalized = (
            self._normalize_text(
                query
            )
        )

        text_normalized = (
            self._normalize_text(
                text
            )
        )

        score = 0.0

        # --------------------------------------------------------------
        # Numbers in the query are often highly discriminative.
        #
        # Example:
        #
        # 2024
        # 500
        # 15
        # --------------------------------------------------------------

        query_numbers = re.findall(
            r"\b\d+\b",
            query_normalized,
        )

        for number in query_numbers:

            if re.search(
                rf"\b{re.escape(number)}\b",
                text_normalized,
            ):

                score += 0.025

        # --------------------------------------------------------------
        # Names / proper nouns.
        #
        # We don't know whether a capitalized word is a person,
        # organization, city, etc. We simply use it as a discriminative
        # token.
        # --------------------------------------------------------------

        original_words = re.findall(
            r"\b[A-Z][a-zA-Z]{2,}\b",
            query,
        )

        for word in original_words:

            normalized_word = (
                self._normalize_text(
                    word
                )
            )

            if (
                normalized_word
                and normalized_word in text_normalized
            ):

                score += 0.025

        return min(
            score,
            0.10,
        )

    # ==================================================================
    # CONTEXT SUFFICIENCY
    # ==================================================================

    def has_sufficient_context(
        self,
        results: list[RetrievalResult],
    ) -> bool:

        if not results:

            print(
                "\n[RAG DEBUG] "
                "CONTEXT CHECK"
            )

            print(
                "No retrieval results."
            )

            return False

        # --------------------------------------------------------------
        # Page-expanded chunks can have zero similarity scores.
        #
        # Therefore we only use actual semantic scores here.
        # --------------------------------------------------------------

        semantic_scores = [
            result.score
            for result in results
            if result.score > 0
        ]

        if not semantic_scores:

            print(
                "\n[RAG DEBUG] "
                "CONTEXT CHECK"
            )

            print(
                "No original semantic scores."
            )

            return False

        best_score = max(
            semantic_scores
        )

        print(
            "\n[RAG DEBUG] "
            "CONTEXT CHECK"
        )

        print(
            f"Best semantic score: "
            f"{best_score}"
        )

        print(
            f"Required threshold: "
            f"{self.score_threshold}"
        )

        sufficient = (
            best_score
            >= self.score_threshold
        )

        print(
            f"Context sufficient: "
            f"{sufficient}"
        )

        return sufficient