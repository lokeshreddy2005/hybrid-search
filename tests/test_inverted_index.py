import tempfile
from pathlib import Path

from index.inverted_index import InvertedIndex

DOCS = [
    ("d1", "", "the cat sat on the mat"),
    ("d2", "", "the dog sat on the log"),
    ("d3", "", "cats and dogs are great pets"),
]


def build_index() -> InvertedIndex:
    idx = InvertedIndex()
    idx.build(DOCS)
    return idx


def test_doc_count_and_ids():
    idx = build_index()
    assert idx.n_docs == 3
    assert idx.doc_ids == ["d1", "d2", "d3"]


def test_doc_lengths_after_stopword_removal():
    idx = build_index()
    # d1: "the cat sat on the mat" -> cat, sat, mat (3 tokens)
    # d2: "the dog sat on the log" -> dog, sat, log (3 tokens)
    # d3: "cats and dogs are great pets" -> cats, dogs, great, pets (4 tokens)
    assert idx.doc_lengths == [3, 3, 4]
    assert idx.avg_doc_length == (3 + 3 + 4) / 3


def test_postings_list_correctness():
    idx = build_index()
    # "sat" appears in d1 (internal id 0) and d2 (internal id 1), tf=1 each
    assert idx.get_postings("sat") == [(0, 1), (1, 1)]
    # "cat" only appears in d1
    assert idx.get_postings("cat") == [(0, 1)]
    # term not in corpus
    assert idx.get_postings("nonexistent") == []


def test_doc_freq():
    idx = build_index()
    assert idx.doc_freq("sat") == 2
    assert idx.doc_freq("cat") == 1
    assert idx.doc_freq("nonexistent") == 0


def test_term_frequency_within_doc():
    idx = InvertedIndex()
    idx.build([("d1", "", "cat cat dog cat")])
    # "cat" appears 3 times in d1 (internal id 0)
    postings = idx.get_postings("cat")
    assert postings == [(0, 3)]
    postings_dog = idx.get_postings("dog")
    assert postings_dog == [(0, 1)]


def test_save_and_load_roundtrip():
    idx = build_index()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "idx.pkl"
        idx.save(path)
        loaded = InvertedIndex.load(path)
        assert loaded.doc_ids == idx.doc_ids
        assert loaded.postings == idx.postings
        assert loaded.doc_lengths == idx.doc_lengths
