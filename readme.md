*This project has been created as part of the 42 curriculum by ael-bala.*

# RAG against the machine

A Retrieval-Augmented Generation (RAG) system that answers questions about the [vLLM](https://github.com/vllm-project/vllm) v0.10.1 codebase. It indexes the repository, retrieves the most relevant snippets for a question, and has a small local model (`Qwen/Qwen3-0.6B`) write an answer grounded in them.

> **Before submitting:** replace every `XX`, `login` and `TODO` below with your own real values, and make sure each statement matches what your code actually does. You will be asked to explain and defend all of it.

---

## Description

### Goal

A language model only knows what it was trained on. Instead of retraining it, RAG gives it access to an external source (here, the vLLM repository) and lets it pull the right information at answer time.

This project builds the full pipeline:

1. **Indexing**: read the repository, split every file into chunks, and persist a searchable index.
2. **Retrieving**: for a question, return the top-k most relevant chunks as source locations (`file_path` + character range).
3. **Augmenting**: place the retrieved snippets in the model's context window within its token budget.
4. **Generating**: produce a coherent, grounded, on-point answer with `Qwen/Qwen3-0.6B`.

Retrieval quality is measured with **recall@k**: a correct source counts as found when one of the retrieved results is in the same file and overlaps the reference character range (IoU ≥ 0.05).

### Overview

| Stage | Implementation |
|---|---|
| Chunking | Two strategies: Python code and Markdown/text |
| Retrieval | BM25 (TF-IDF as an alternative) |
| Generation | `Qwen/Qwen3-0.6B`, CPU, structured JSON output |
| Data validation | `pydantic` models |
| CLI | Python Fire + `tqdm` progress bars |
| Tooling | `uv`, `flake8`, `mypy` |

---

## Instructions

### Requirements

- Python 3.10 or later
- [`uv`](https://docs.astral.sh/uv/) as the project and package manager
- Several GB of free disk space (deep-learning dependencies and model weights)
- The vLLM v0.10.1 repository placed under `data/raw/`

### Installation

```bash
git clone <repository-url>
cd <repository>
uv sync          # or: make install
```

### Expected layout

```
.
├── src/                          # Python module (uv run python -m src <command>)
├── data/
│   ├── raw/                      # indexed sources (vLLM repository)
│   ├── processed/                # index produced by `index`
│   ├── datasets/
│   │   ├── UnansweredQuestions/  # questions only
│   │   └── AnsweredQuestions/    # ground truth
│   └── output/
│       ├── search_results/<DatasetScope>/
│       └── search_results_and_answer/<DatasetScope>/
├── pyproject.toml
├── uv.lock
├── Makefile
└── README.md
```

### Makefile rules

| Rule | Description |
|---|---|
| `make install` | Install dependencies with `uv sync` |
| `make run` | Run the main script |
| `make debug` | Run the main script under `pdb` |
| `make clean` | Remove `__pycache__`, `.mypy_cache`, and other caches |
| `make lint` | `flake8 .` and `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs` |
| `make lint-strict` | `flake8 .` and `mypy . --strict` (optional) |

### CLI commands

All commands are run as `uv run python -m src <command> [options]`. Every input and output path is a CLI argument; nothing is hard-coded.

| Command | Purpose |
|---|---|
| `index --max_chunk_size <int>` | Ingest `data/raw/` and build the index under `data/processed/` |
| `search <query> --k <int>` | Top-k sources for a single query |
| `search_dataset --dataset_path <path> --k <int> --save_directory <dir>` | Run search over a dataset and write a `StudentSearchResults` JSON |
| `answer <query> --k <int>` | Answer a single query from the retrieved context |
| `answer_dataset --student_search_results_path <path> --save_directory <dir>` | Generate answers for a dataset and write a `StudentSearchResultsAndAnswer` JSON |
| `evaluate --student_search_results_path <path> --dataset_path <path>` | Compute recall@k against a ground-truth dataset (for local testing only) |

Degenerate inputs (empty query, `k=0`, missing file, malformed JSON) are handled with a clear error message instead of a traceback.

---

## Example usage

### 1. Build the index

```bash
uv run python -m src index --max_chunk_size 2000
# Ingestion complete! Indexed N chunks under data/processed/
```

### 2. Search a single question

```bash
uv run python -m src search "How to configure the OpenAI server?" --k 5
# data/raw/vllm-0.10.1/examples/online_serving/openai_chat_completion_client.py [0:1352]
# data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md [1936:3170]
# ...
```

### 3. Search a whole dataset

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

### 4. Score the retrieval

With the provided moulinette (rename `moulinette-ubuntu` / `moulinette-fedora` to `moulinette` first):

```bash
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000
```

Or with the built-in command, for quick iteration:

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

### 5. Generate answers

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

### 6. Answer a single question

```bash
uv run python -m src answer "How do I enable LoRA adapters?" --k 5
```

---

## System architecture

```
 data/raw/ ──► Chunking ──► Index (BM25) ──► data/processed/
 (vLLM repo)   (code | md)                          │
                                                    ▼
 question ──► Tokenize ──► Retrieve top-k ──► Augment (build prompt) ──► Qwen3-0.6B ──► JSON answer
                                  │
                                  └──► Evaluate (recall@k, IoU > 0.05)
```

| Component | Module | Responsibility |
|---|---|---|
| CLI | `src/__main__.py` | Python Fire entry point, argument handling, error handling |
| Data models | `src/models.py` | `pydantic` models (`MinimalSource`, `UnansweredQuestion`, `AnsweredQuestion`, `RagDataset`, `MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`) |
| Chunking | `src/chunking.py` | Python and Markdown/text chunkers |
| Indexer | `src/indexer.py` | Reads the corpus, chunks it, builds and persists the index |
| Retriever | `src/retriever.py` | Loads the index once, scores queries, returns ranked sources |
| Generator | `src/generator.py` | Builds the prompt from retrieved context, runs Qwen, parses the output |
| Pipeline | `src/pipeline.py` | Orchestrates single-query and dataset-level commands |
| Evaluation | `src/evaluate.py` | Local recall@k computation |

> **TODO:** adjust file and module names to match your actual layout.

The index is built once by `index` and loaded once by every other command, so a whole dataset is searched without rebuilding or re-reading anything per question.

---

## Chunking strategy

A Python file and a Markdown page do not break apart the same way, so the indexer uses two strategies, selected by file extension. Every chunk is **at most `--max_chunk_size` characters** (default 2000, never above), and each chunk keeps its `file_path` and exact character offsets so results map back to the source file.

### Python code

- Parse the file (for example with `ast`) and split on top-level boundaries: functions, classes, and methods.
- Keep decorators and docstrings attached to their definition.
- Keep the module header (imports, module docstring) as its own chunk.
- If a definition is larger than the maximum size, split it further on blank lines or line boundaries, never in the middle of a line.
- If parsing fails, fall back to line-based splitting.

### Markdown / text

- Split on heading boundaries so each chunk is a coherent section.
- Keep the heading text with its section (useful for matching paraphrased questions).
- Sections larger than the maximum size are split on paragraph boundaries, then on sentence or line boundaries.
- Fenced code blocks are not split in the middle where it can be avoided.

### Effect of chunk size on recall@5

| `--max_chunk_size` | Docs recall@5 | Code recall@5 | Indexing time |
|---|---|---|---|
| 2000 | XX.X % | XX.X % | XX s |
| 1500 | XX.X % | XX.X % | XX s |
| 1000 | XX.X % | XX.X % | XX s |
| 500 | XX.X % | XX.X % | XX s |

> **TODO:** fill in your measured values and add one or two sentences on what you observed (for example, smaller chunks give more precise spans but split the answer across several chunks).

---

## Retrieval method

### Algorithm: BM25

Each chunk is scored against the query with Okapi BM25:

```
score(q, d) = Σ over t in q:  IDF(t) · f(t, d) · (k1 + 1) / ( f(t, d) + k1 · (1 − b + b · |d| / avgdl) )

IDF(t) = ln( 1 + (N − n(t) + 0.5) / (n(t) + 0.5) )
```

- `f(t, d)`: frequency of term `t` in chunk `d`
- `|d|`, `avgdl`: chunk length and average chunk length (in tokens)
- `N`, `n(t)`: total number of chunks and number of chunks containing `t`
- `k1 = XX`, `b = XX` (tuned on the public datasets)

The top-k chunks by score are returned as `MinimalSource` objects (`file_path`, `first_character_index`, `last_character_index`).

### Tokenization

Questions and source code rarely use the same words, so tokenization is tuned for code:

- Lowercase everything.
- Split `snake_case`, `camelCase`, and dotted paths into sub-tokens (`max_chunk_size` → `max`, `chunk`, `size`).
- Keep the full identifier as an additional token, so a question quoting an identifier verbatim still matches exactly.
- Remove stop words for prose; keep short tokens that are meaningful in code.
- Index the file path (and Markdown headings) so questions about a file or feature can match its location.

### Ranking and output

- Scores are computed over all chunks and sorted in descending order.
- Results are deduplicated so the top-k are not several near-identical chunks of the same region.
- `file_path` is written exactly as in the ingested corpus (`data/raw/vllm-0.10.1/...`), because the grader compares paths verbatim.

> **TODO:** if you implemented TF-IDF instead of (or in addition to) BM25, describe it here, and describe any extras (hybrid retrieval, reranking, query expansion).

---

## Answer generation

- **Model:** `Qwen/Qwen3-0.6B`, running on CPU.
- **Context building:** the top retrieved chunks are read back from `data/raw/` (or from the index), then truncated to fit the model's token budget.
- **Prompt:** the model is instructed to answer only from the provided context, to stay on the question asked, and to say so when the context does not contain the answer.
- **Thinking mode:** disabled, so the output is a clean answer rather than reasoning text.
- **Output:** a `MinimalAnswer` per question, collected in a `StudentSearchResultsAndAnswer` file.
- **Failures:** a question without sources, or a generation error, produces a fallback answer instead of stopping the batch.

---

## Performance analysis

### Recall@k

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Required (recall@5) |
|---|---|---|---|---|---|
| Docs (public) | XX.X % | XX.X % | XX.X % | XX.X % | ≥ 80 % |
| Code (public) | XX.X % | XX.X % | XX.X % | XX.X % | ≥ 50 % |

### System performance

| Metric | Measured | Required |
|---|---|---|
| Indexing time (whole corpus) | XX s | ≤ 5 min |
| Retrieval, 200 questions | XX s | ≤ 90 s |
| Number of chunks indexed | XX | n/a |
| Answer generation, per question | XX s | n/a |

> **TODO:** note the machine used for these measurements, and discuss where recall is lost (for example, which question types fail and why).

---

## Design decisions

- **BM25 over TF-IDF:** BM25 handles document length and term saturation better on chunks of very different sizes. *(Adapt to your own reasoning.)*
- **Two chunkers:** structure-aware splitting keeps functions and sections intact, which gives better overlap with the reference spans than fixed-size windows.
- **Code-aware tokenization:** splitting identifiers while keeping the original token lets us match both paraphrased and verbatim questions.
- **Persisted index, loaded once:** keeps indexing separate from querying, so searching 200 questions stays well under the time limit.
- **Hard cap on chunk size:** the cap is enforced in the chunkers themselves, so no result can exceed the grader's `max_context_length`.
- **`pydantic` for data models:** all data exchanged between stages is validated, which catches malformed datasets early.
- **Search and answer are separate commands:** retrieval can be scored on its own before running the slower generation step.

---

## Challenges faced

> **TODO:** replace with your own experience. Likely topics:

- **Exact file paths:** the grader compares `file_path` verbatim, so the corpus path prefix must be preserved from indexing through to the output.
- **Chunk size limit:** a single source over 2000 characters invalidates the whole output, so the cap is enforced at chunk creation and checked on output.
- **Vocabulary mismatch between questions and code:** solved with identifier splitting while keeping the original token.
- **Indexing time:** keeping the whole corpus under 5 minutes required (describe your optimizations).
- **Small model limits:** `Qwen3-0.6B` has a small effective context and weak reasoning, so the prompt and the amount of context are kept short and focused.
- **Robustness:** handling empty queries, `k=0`, missing files, and malformed JSON without a traceback.

---

## Resources

### References

- [vLLM documentation](https://docs.vllm.ai/)
- [vLLM repository](https://github.com/vllm-project/vllm)
- [Qwen3 model card (`Qwen/Qwen3-0.6B`)](https://huggingface.co/Qwen/Qwen3-0.6B)
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Python Fire](https://github.com/google/python-fire)
- [uv documentation](https://docs.astral.sh/uv/)

### Use of AI

> **TODO:** describe honestly which tasks and which parts of the project you used AI for, and how you verified the result. Example format:

- *Concepts:* used AI to understand BM25 and chunking trade-offs, then validated against the documentation.
- *Boilerplate:* used AI for the Makefile and the pydantic model skeletons, then reviewed and tested them.
- *Debugging:* used AI to interpret error messages; the fixes were written and understood by us.
- *Not used for:* (list the parts you wrote yourself, such as the chunkers and the retriever).

Everything produced with AI assistance was reviewed, tested, and discussed with peers, and can be explained by the authors.