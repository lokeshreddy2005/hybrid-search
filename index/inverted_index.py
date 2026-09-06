"""A from-scratch inverted index: term -> postings list of (doc_id, term_freq).

doc_id here is an *internal* integer (0..N-1) assigned in insertion order.
Callers that need external/original ids keep a parallel `doc_ids` list.
"""
from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from index.tokenizer import tokenize


@dataclass
class InvertedIndex:
    # term -> list of (internal_doc_id, term_frequency), sorted by doc_id
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    doc_lengths: list[int] = field(default_factory=list)   # tokens per doc, by internal id
    doc_ids: list[str] = field(default_factory=list)        # internal id -> external id
    doc_titles: list[str] = field(default_factory=list)     # internal id -> title (for display)

    @property
    def n_docs(self) -> int:
        return len(self.doc_lengths)

    @property
    def avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths) / len(self.doc_lengths)

    def doc_freq(self, term: str) -> int:
        """Number of documents containing `term`."""
        return len(self.postings.get(term, ()))

    def get_postings(self, term: str) -> list[tuple[int, int]]:
        return self.postings.get(term, [])

    def build(self, documents: list[tuple[str, str, str]]) -> None:
        """Build the index from (doc_id, title, text) triples.

        Text is tokenized and term frequencies are counted per document;
        a document contributes at most one (doc_id, tf) entry per term,
        appended in doc-insertion order so each postings list stays
        sorted by internal doc_id without an extra sort pass.
        """
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for internal_id, (doc_id, title, text) in enumerate(documents):
            full_text = f"{title} {text}" if title else text
            tokens = tokenize(full_text)
            self.doc_ids.append(doc_id)
            self.doc_titles.append(title)
            self.doc_lengths.append(len(tokens))

            term_counts = Counter(tokens)
            for term, tf in term_counts.items():
                postings[term].append((internal_id, tf))

        self.postings = dict(postings)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "InvertedIndex":
        with open(path, "rb") as f:
            return pickle.load(f)
