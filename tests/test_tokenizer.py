from index.tokenizer import tokenize


def test_lowercases():
    assert tokenize("Hello WORLD") == ["hello", "world"]


def test_strips_punctuation():
    assert tokenize("Hello, world! It's a test.") == ["hello", "world", "test"]


def test_removes_stopwords_by_default():
    tokens = tokenize("the cat is on the mat")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "on" not in tokens
    assert tokens == ["cat", "mat"]


def test_keep_stopwords_when_disabled():
    tokens = tokenize("the cat is on the mat", remove_stopwords=False)
    assert tokens == ["the", "cat", "is", "on", "the", "mat"]


def test_keeps_numbers():
    assert tokenize("The year 1969 was pivotal") == ["year", "1969", "pivotal"]


def test_empty_string():
    assert tokenize("") == []


def test_keeps_contractions_as_single_token():
    tokens = tokenize("don't stop", remove_stopwords=False)
    assert "don't" in tokens
