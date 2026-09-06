"""Tokenization for the inverted index.

Deliberately dependency-free: a regex word splitter, lowercasing, and a
built-in English stopword list. No stemming — see README limitations.
"""
import re

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Standard-ish English stopword list (subset of the SMART/NLTK list).
# Kept in-repo rather than pulled from NLTK so the index build has no
# extra runtime dependency.
STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll
he's her here here's hers herself him himself his how how's i i'd i'll
i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours
ourselves out over own same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them themselves
then there there's these they they'd they'll they're they've this those
through to too under until up very was wasn't we we'd we'll we're we've
were weren't what what's when when's where where's which while who
who's whom why why's with won't would wouldn't you you'd you'll you're
you've your yours yourself yourselves is
""".split())


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Lowercase, split into word tokens, optionally drop stopwords.

    Numbers are kept (e.g. "1969") since they matter for factual queries
    over an encyclopedic corpus. Apostrophes inside words are preserved
    ("don't") then the whole token is looked up in STOPWORDS.
    """
    if not text:
        return []
    tokens = _WORD_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
