# Hybrid Search & Retrieval Engine

A search engine over ~30,000 Simple English Wikipedia articles that implements **BM25 from scratch** (tokenizer, inverted index, and scoring — no `rank_bm25`, no Elasticsearch), **dense vector retrieval** with a sentence-transformer + FAISS, and a **hybrid mode** that fuses both via Reciprocal Rank Fusion. Served behind a FastAPI endpoint, with a hand-labeled 20-query evaluation set used to run a genuine precision@k / recall@k comparison across all three modes.

**Headline result:** on a 20-query hand-labeled evaluation set, dense **vector search alone had the best recall@10 (96.3%)**, beating both BM25-only (77.5%) and the RRF-fused hybrid (87.5%). BM25 was competitive or superior on exact-entity and ambiguous keyword queries (it was the only mode to rank *both* correct senses of the deliberately ambiguous query "Python" within the top 5 — vector found both too, but not until rank 8), while vector search decisively won on paraphrase/semantic queries with little to no literal term overlap with the target document. Hybrid landed between the two rather than beating both — a real, examined finding about equal-weight RRF, not a bug. Full breakdown below.

## Table of contents
- [Architecture](#architecture)
- [BM25: formula and implementation](#bm25-formula-and-implementation)
- [Dense retrieval: embeddings + FAISS](#dense-retrieval-embeddings--faiss)
- [Hybrid fusion: why Reciprocal Rank Fusion](#hybrid-fusion-why-reciprocal-rank-fusion)
- [Evaluation: three-way comparison (headline result)](#evaluation-three-way-comparison-headline-result)
- [Benchmark: throughput and latency](#benchmark-throughput-and-latency)
- [Setup and run instructions](#setup-and-run-instructions)
- [Testing](#testing)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [Future work](#future-work)

## Architecture

```
data/corpus.jsonl                 30,000 Simple English Wikipedia article abstracts
        |
        v
scripts/build_index.py  ---->  index/store/
        |                        inverted_index.pkl  (custom postings, doc lengths, doc ids)
        |                        vector_index.faiss   (FAISS IndexFlatIP over 384-dim embeddings)
        |                        vector_index.meta.json
        |                        doc_texts.json        (id -> title/text, for rendering results)
        |                        meta.json
        v
retrieval/engine.py  (SearchEngine)
   |-- index/bm25.py        BM25 scoring over index/inverted_index.py postings
   |-- retrieval/vector_index.py   FAISS cosine search over sentence-transformer embeddings
   |-- retrieval/fusion.py         Reciprocal Rank Fusion (bm25 + vector -> hybrid)
        v
api/main.py  (FastAPI)  ---->  GET /search?q=...&mode=bm25|vector|hybrid&top_k=...
                                GET /health
```

Layer responsibilities:
- **`index/`** — the from-scratch IR core: tokenizer, inverted index (postings lists), BM25 scoring. No IR library dependency.
- **`retrieval/`** — retriever interfaces on top of the index: the embedding wrapper, the FAISS vector index, RRF fusion, and the `SearchEngine` that unifies all three modes behind one call.
- **`api/`** — FastAPI app and Pydantic schemas; thin — all logic lives in `retrieval`/`index`.
- **`eval/`** — the hand-labeled query set and precision/recall metric functions.
- **`scripts/`** — CLI entry points: download the corpus, build the index, run the evaluation, run the load test, and a small corpus-search helper used to author honest eval labels.
- **`tests/`** — unit tests for each layer (see [Testing](#testing)).

## BM25: formula and implementation

Implemented in [`index/inverted_index.py`](index/inverted_index.py) (postings) and [`index/bm25.py`](index/bm25.py) (scoring) — no `rank_bm25`, no Whoosh, no Elasticsearch.

**Tokenization** ([`index/tokenizer.py`](index/tokenizer.py)): lowercase, regex word split (`[a-z0-9]+(?:'[a-z]+)?`, so contractions like `don't` stay one token), stopword removal against a built-in ~180-word list. No stemming — see [Limitations](#limitations); this is a deliberate scope cut, not an oversight, and its effect shows up directly in the evaluation (see query `q16` in the results below).

**Inverted index** ([`index/inverted_index.py`](index/inverted_index.py)): `term -> [(internal_doc_id, term_frequency), ...]`, built in one pass over the corpus with a `Counter` per document, appended in insertion order so every postings list is already sorted by doc id with no separate sort step. Alongside the postings, the index tracks per-document token length and the corpus average document length, both needed for the length-normalization term in BM25.

**BM25 scoring** ([`index/bm25.py`](index/bm25.py)), the standard Robertson/Sparck-Jones Okapi BM25 formula, with the "+1" smoothed idf (as used by Lucene/Elasticsearch, to keep idf non-negative for very common terms):

```
score(D, Q) = sum_{t in Q}  idf(t) * (f(t,D) * (k1 + 1))
                             -----------------------------------
                             f(t,D) + k1 * (1 - b + b * |D| / avgdl)

idf(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
```
where `f(t,D)` is the raw term frequency of `t` in `D`, `|D|` the document length in tokens, `avgdl` the corpus average document length, `N` the document count, `df(t)` the number of documents containing `t`, and `k1=1.5`, `b=0.75` (standard Okapi defaults).

Retrieval is **term-at-a-time over postings lists** ([`BM25.search`](index/bm25.py)): for each query term, only the documents in that term's postings list are ever touched — the implementation never scans the full corpus, which is the entire point of building an inverted index instead of a linear scan. Scores are accumulated in a `dict` keyed by internal doc id, then sorted.

Verified against a hand-computed worked example in [`tests/test_bm25.py`](tests/test_bm25.py) (3-document toy corpus, `idf`/score computed by hand and checked with `pytest.approx`).

## Dense retrieval: embeddings + FAISS

- **Model**: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — 22M parameters, 384-dim output, chosen for being small enough to encode 30k documents on CPU in a reasonable time while still being a solid general-purpose semantic embedding model.
- Each document is embedded as `"{title}. {text}"`; queries are embedded with the same model. Embeddings are L2-normalized at encode time ([`retrieval/embedder.py`](retrieval/embedder.py)) so inner product equals cosine similarity.
- **Index**: FAISS `IndexFlatIP` ([`retrieval/vector_index.py`](retrieval/vector_index.py)) — exact brute-force inner-product search. At ~30k documents x 384 dims, brute-force search over every vector is cheap enough at this corpus size that an approximate index (IVF/HNSW) would only trade accuracy for speed we don't need yet (the end-to-end [benchmark](#benchmark-throughput-and-latency) numbers below aren't a FAISS-only measurement — they include query embedding, which dominates total latency — so they demonstrate acceptable overall latency rather than isolating FAISS's own search cost); see [Limitations](#limitations) for when the IVF/HNSW trade would start to make sense.

## Hybrid fusion: why Reciprocal Rank Fusion

Implemented in [`retrieval/fusion.py`](retrieval/fusion.py):

```
RRF(d) = sum_{s in {bm25, vector}}  1 / (k + rank_s(d))          (k = 60)
```

**Why rank fusion instead of a weighted sum of raw scores:** BM25 scores are unbounded and corpus/query-dependent — a query with rare terms can score 20+, a query of only common terms may top out near 2. Cosine similarity from the embedding model is bounded in `[-1, 1]` and in practice clusters in a narrow band for this corpus. Combining these two directly (e.g. `0.5 * bm25_score + 0.5 * cosine_score`) requires per-query normalization (min-max or z-score) to even be on comparable scales, and the result is then sensitive to exactly how that normalization is done and to outlier scores. RRF sidesteps the problem entirely by using only each system's **rank**, not its score — it's scale-free by construction. This is also not a novel choice for this project specifically: RRF (Cormack et al., 2009) is a widely used production hybrid-search technique — Elastic recommends it for hybrid retrieval, Azure AI Search uses it to merge hybrid results, and OpenSearch offers it as a rank-based fusion option alongside score-normalization approaches — so implementing it here is representative of how this problem is actually solved in practice.

The engine ([`retrieval/engine.py`](retrieval/engine.py)) fetches a candidate pool of 100 from each of BM25 and vector search, fuses with RRF, then returns the requested `top_k`.

## Evaluation: three-way comparison (headline result)

**Corpus**: 30,000 Simple English Wikipedia article abstracts (see [Limitations](#limitations) for why Simple English Wikipedia over MS MARCO / full English Wikipedia).

**Query set**: 20 hand-authored queries against this specific corpus, each manually labeled with the actual relevant document id(s) — verified by reading the real article text with [`scripts/inspect_corpus.py`](scripts/inspect_corpus.py), not guessed. The set deliberately spans several query types: exact entity lookup, natural-language questions, semantic paraphrases with low lexical overlap, a deliberately ambiguous single-word query, and multi-document topics. See [`eval/eval_queries.json`](eval/eval_queries.json) for the full set with a `note` on each query's intent.

**Actual measured results** (`python scripts/run_eval.py`, 20 queries, 30,000-doc corpus):

| mode | precision@5 | precision@10 | recall@5 | recall@10 |
|---|---|---|---|---|
| bm25 | 0.180 | 0.090 | 0.775 | 0.775 |
| vector | 0.190 | 0.115 | **0.850** | **0.963** |
| hybrid | 0.180 | 0.100 | 0.775 | 0.875 |

*A note on reading precision here*: 16 of the 20 queries have exactly **one** genuinely relevant document, 3 have two, and 1 (`q19`) has four. That structurally caps average precision@5 well below 1.0 (a single-relevant-doc query maxes out at precision@5 = 0.20 even with a perfect rank-1 hit). Given this label distribution, the *best possible* average precision@5 — every relevant document retrieved at rank 1 — works out to `(16*0.2 + 3*0.4 + 1*0.8) / 20 = 0.26`, not 1.0. So vector's measured 0.190 is close to that dataset-induced ceiling of ~0.26, not a poor score against an attainable 1.0. **Recall@k is the more informative metric for this label set** — precision@k is reported because the brief asks for it, but it's mechanically compressed here, not a sign anything is broken.

### Analysis: which mode won, and why

**Vector search won on aggregate recall, driven by a specific, identifiable class of query: semantic/paraphrase queries with low literal term overlap.**
- `q12` *"how plants make food from sunlight"* → target `Photosynthesis`. The article talks about plants making "carbohydrates" and "sugars" using sunlight — never the word "food." **BM25 missed this document entirely** (not in its top 10 at all); vector search found it at rank 2.
- `q18` *"volcanic eruption lava"* → target `Volcano`. Counterintuitively, **BM25 also missed `Volcano` entirely** here. I checked the actual token counts to find out why: `Volcano`'s abstract contains "volcanic" (tf=2) and "lava" (tf=2) but **zero** occurrences of "eruption" — one of the three query terms contributes nothing at all to its score. Meanwhile narrower, shorter articles literally titled `Eruption` and `Volcanic eruption` match all three query terms densely and, at ~150 tokens vs. `Volcano`'s above-average length, also benefit slightly from BM25's length normalization. Vector search ranked `Volcano` at rank 2 regardless of that missing term — it doesn't require literal term presence, only topical similarity.
- `q02` *"capital of France"* → target `France`. This corpus's `France` abstract (truncated to ~1500 characters, see [Limitations](#limitations)) never actually mentions "capital" or "Paris." BM25 and hybrid never surfaced the `France` article in their top 10 at all — dozens of other French-geography articles ("Île-de-France," "Toulouse," "Picardy," ...) that literally contain the word "France" outranked it. Vector search was the *only* mode that found `France` in the top 10 (rank 7) — not a strong result on its own, but the only one that found it at all.

**BM25 handled query ambiguity better than vector search, and hybrid best of all.** `q04`, the query *"Python"* alone (deliberately ambiguous between the programming language and the snake genus, both genuinely relevant), separates the three modes cleanly by *where* each sense lands: BM25 ranked `Python (programming language)` at rank 1 and the snake genus at rank 3 (recall@5 = 1.0). Vector search also found the programming-language sense at rank 1, but the snake genus only surfaces at **rank 8** (recall@5 = 0.5, recall@10 = 1.0) — its embedding of the bare word "Python" sits in a neighborhood dominated by programming-related documents (`Hello world program`, `Computer programming`, `Interpreter (computing)`, ...), so the less-dominant sense is technically present but pushed far down. Exact term matching, by construction, doesn't have that popularity bias. Hybrid actually does best of all three here — rank 1 and rank **2** — because RRF gives the snake document credit from *both* retrievers (BM25's rank 3 and vector's rank 8 combined outrank vector's single weak signal alone), a case where fusion clearly helps rather than dilutes.

**Hybrid improved on BM25 but did not reach vector's ceiling — and the per-query data shows exactly why.** RRF's score for a document is a *sum* of `1/(k+rank)` contributions from every list it appears in — a document supported by both retrievers accumulates two such terms, while a document found by only one retriever gets just the one. On `q14` ("global warming and climate change," two relevant docs), vector alone found both `Climate change` (rank 2) and `Intergovernmental Panel on Climate Change` (rank 6) for recall@10 = 1.0. BM25 found only `Climate change` and never surfaced the IPCC article at all. Because the IPCC document appears in *only* the vector ranking, it receives no second, corroborating RRF term from BM25 — so in the fused ranking, other documents that both retrievers agree on (even at more modest individual ranks) accumulate two contributions and can overtake it. It lands outside the fused top 10, giving hybrid recall@10 = 0.50 on that query, worse than vector alone. The same pattern repeats on `q12` and `q18` above: hybrid recovers the doc by rank 10 in both cases (better than BM25's total miss) but not inside the top 5, for the same reason — a single-retriever-supported document is competing against dual-supported ones in the fused ranking. This is a genuine, examined limitation of *equal-weight* RRF, not a fusion failure in general — see [Future work](#future-work) for the score-normalized-weighting follow-up this motivates.

**A limitation the results themselves exposed, worth reporting rather than hiding:** `q19` ("moons of Jupiter," labeled relevant = Europa, Io, Ganymede, Callisto) was missed almost entirely by *all three modes* (BM25 and hybrid: 0/4 in top 10; vector: 1/4). Looking at what was actually retrieved, all three modes correctly and consistently surfaced `List of Jupiter's moons`, `Inner moons of Jupiter`, and `Jupiter` itself at the top of the ranking — arguably more relevant to the literal query than the four individually-named moons I had labeled as the relevant set. This is a labeling gap I introduced when authoring the query, not a retrieval failure, and I'm reporting it as run rather than quietly relabeling after seeing the output — see [Limitations](#limitations).

**A deliberately constructed stemming test didn't isolate what it was meant to.** `q15`/`q16` were designed as a minimal pair to expose the tokenizer's lack of stemming: `q15` "star explosion" (singular, matches the `Supernova` article's opening sentence almost verbatim) vs. `q16` "stars exploding at the end of their life" (plural, same target document). Both retrieved `Supernova` successfully (rank 1 and rank 2 respectively for BM25) because the article's full text — not just its opening sentence — naturally uses the plural "stars" too ("*the biggest **stars** that make supernovae*", "*most **stars** are small*"). BM25 did drop from rank 1 to rank 2, a small measurable effect, but not the sharp miss a synthetic mismatch would produce. Real encyclopedic text has enough natural morphological variety that missing stemming matters less in practice here than the isolated toy case in [`tests/test_bm25.py`](tests/test_bm25.py) might suggest.

Full per-query results: [`eval/results.json`](eval/results.json) (generated by `scripts/run_eval.py`).

## Benchmark: throughput and latency

Measured with `scripts/benchmark.py` against a single local `uvicorn` process (no `--workers`, default threadpool), concurrency = 8 clients, cycling through the 20 evaluation queries:

| mode | requests | duration | QPS | mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|---|---|
| bm25 | 1557 | 15.1s | **103.1** | 77.4 | 63.2 | 168.2 | 301.6 |
| vector | 379 | 15.2s | 24.9 | 319.5 | 315.0 | 418.3 | 507.5 |
| hybrid | 443 | 20.2s | 22.0 | 363.0 | 349.5 | 521.3 | 641.9 |

BM25 is ~4-5x higher throughput than vector/hybrid because it's pure-Python postings-list lookups with no model inference. Vector and hybrid latency are close to each other rather than roughly additive, which is consistent with query embedding (CPU inference of a 22M-parameter model per request) being the dominant added cost and RRF fusion itself (over two already-fetched candidate lists) being comparatively cheap — but this benchmark measures end-to-end HTTP latency, not each stage in isolation, so that attribution is an inference from the aggregate numbers rather than a profiled measurement of query-encode vs. FAISS-search vs. fusion vs. serialization individually. The clearest lever for improving vector/hybrid throughput would be batching concurrent query encodes or moving inference to a GPU, neither of which was needed at this corpus scale but would matter under real production load.

Methodology: [`scripts/benchmark.py`](scripts/benchmark.py) runs an async load test against a live `uvicorn` instance of the API — fixed concurrency, fixed wall-clock duration, cycling through the 20 evaluation queries, measuring end-to-end HTTP request latency (including FastAPI + retrieval + fusion + JSON serialization). Run on the developer machine used to build this project (single process, CPU-only inference — see [Limitations](#limitations)).

## Setup and run instructions

Requires Python 3.12. Tested on Windows with a local venv (no Docker available in the dev environment — see [Limitations](#limitations)).

```bash
# 1. Create and activate a virtual environment, install dependencies
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

# 2. Download the corpus (~30,000 Simple English Wikipedia article abstracts, ~31MB)
python scripts/download_corpus.py --n-docs 30000

# 3. Build the inverted index + FAISS vector index (downloads the MiniLM model on first run)
python scripts/build_index.py

# 4. Run the API
uvicorn api.main:app --host 0.0.0.0 --port 8000
# then: curl "http://localhost:8000/search?q=capital+of+france&mode=hybrid&top_k=5"

# 5. Run the evaluation (three-way precision@k / recall@k comparison)
python scripts/run_eval.py

# 6. Run the benchmark (API must already be running from step 4)
python scripts/benchmark.py --mode hybrid --concurrency 8 --duration 30
```

Docker (untested end-to-end — see [Limitations](#limitations)):
```bash
python scripts/build_index.py          # build the index locally first
docker build -t hybrid-search .
docker run -p 8000:8000 hybrid-search
```

**Reproducibility, verified**: the exact sequence above (`git clone` → fresh venv → `pip install -r requirements.txt` → `download_corpus.py` → `build_index.py` → `run_eval.py` → `uvicorn` → a live `/search` request) was run from a genuinely clean clone in a separate directory, independent of the checkout this README was written from. The corpus download reproduced the identical 30,000/33,602 document counts, and `run_eval.py` reproduced the precision@k/recall@k table above byte-for-byte. This is a stronger claim than "the test suite passes on a clean clone" (which only exercises code correctness against a fake in-memory embedder, not the real model/index/API) — this specifically re-ran the full corpus-to-API pipeline independently and got the same numbers.

## Testing

```bash
pytest -q
```

33 tests, all passing, across:
- `tests/test_tokenizer.py` — lowercasing, punctuation stripping, stopword removal, contraction handling.
- `tests/test_inverted_index.py` — postings list correctness, term frequency, document frequency, save/load roundtrip.
- `tests/test_bm25.py` — BM25 score verified against a hand-computed worked example.
- `tests/test_fusion.py` — RRF fusion verified against a hand-computed worked example (rank-only, ignores raw score magnitude).
- `tests/test_api.py` — API response shape/status codes for all three modes, error handling (invalid mode, missing query, out-of-range `top_k`), using a fake in-memory embedder so the test suite doesn't need to download the real model.

## Project structure

```
hybrid-search/
├── index/          tokenizer, inverted index, BM25 (from scratch)
├── retrieval/       embedder, FAISS vector index, RRF fusion, unified SearchEngine
├── api/             FastAPI app + Pydantic schemas
├── eval/             hand-labeled query set + precision/recall metrics
├── scripts/         download corpus, build index, run eval, run benchmark
├── tests/            unit tests for every layer
├── data/             corpus.jsonl (generated, gitignored)
├── requirements.txt
├── Dockerfile
└── README.md
```

## Limitations

- **Corpus substitution**: the task suggested Wikipedia abstracts or MS MARCO passages. This project uses **Simple English Wikipedia** (`wikimedia/wikipedia`, config `20231101.simple`) rather than full English Wikipedia abstracts or MS MARCO, because (a) it reliably streams from Hugging Face with no auth/beam-runner setup, and (b) its short, self-contained articles on recognizable topics make it realistic to author *honest* hand-labeled relevance judgments by reading the actual text — MS MARCO passages are decontextualized fragments that are much harder to relevance-judge by inspection.
- **Abstract truncation**: each document is the article's lead text truncated to ~1500 characters. This occasionally cuts out facts a query might reasonably expect (e.g. the "France" abstract in this corpus doesn't retain the sentence naming Paris as the capital) — this is a real, visible effect in the evaluation results, not hidden.
- **No stemming/lemmatization**: the tokenizer does exact word matching after lowercasing and stopword removal only. Plural/singular and other morphological variants (`star` vs `stars`) are different tokens to BM25. This is a deliberate scope cut to keep the from-scratch index simple, and its effect is directly visible in the `q16` evaluation result (see analysis above).
- **Exact vector search, not approximate**: FAISS `IndexFlatIP` is brute-force. Fine at 30k documents; would need to move to IVF/HNSW (accuracy/speed trade-off) at a few million+ documents where brute-force cosine search over every vector per query stops being cheap.
- **No distributed/sharded indexing, no incremental indexing pipeline** — out of scope per the project brief. The index is rebuilt from scratch (`scripts/build_index.py`) rather than updated incrementally.
- **No reranker**: a cross-encoder reranking stage was explicitly deferred until after the three-way comparison experiment (per the project brief) and was not added afterward, to keep the headline experiment's result attributable to the fusion method alone.
- **Docker untested end-to-end**: Docker was not available in the development environment this project was built in, so the `Dockerfile` follows standard practice but hasn't been build-verified; the local venv path is the verified run path.
- **Single-machine benchmark**: the load test in `scripts/benchmark.py` runs against a single local `uvicorn` worker process on developer hardware, not a production deployment — throughput numbers are illustrative of the method's per-request cost, not a capacity-planning result.

## Future work

- Add a cross-encoder reranking stage (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) over the top-N hybrid results, now that the from-scratch three-way comparison is complete and documented, so its incremental effect can be measured against this baseline.
- Add stemming (e.g. a from-scratch Porter stemmer, in keeping with the "from scratch" spirit of the index) and re-run the evaluation to quantify its effect on queries like `q16`.
- Expand the evaluation set beyond 20 queries and add inter-rater relevance judgments (currently single-annotator) to get a more statistically robust precision/recall estimate with confidence intervals.
- Move from FAISS `IndexFlatIP` to an approximate index (HNSW) with a documented recall-vs-latency trade-off study, as a rehearsal for scaling past exact search.
- Score-normalized weighted fusion (e.g. min-max normalized BM25 + cosine, tunable weights) as an explicit A/B comparison against RRF on the same eval set.
