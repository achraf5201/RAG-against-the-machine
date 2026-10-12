import bm25s
import json
from pathlib import Path

class Rag:
    def __init__(self, chunks, processed_dir="data/processed"):
        self.chunks = chunks
        self.retriever = bm25s.BM25()
        self.processed_dir = Path(processed_dir)

    def indexing(self):
        self.texts = [chunk.page_content for chunk in self.chunks]
        self.corpus_tokens = bm25s.tokenize(self.texts)
        self.retriever.index(self.corpus_tokens)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.retriever.save(self.processed_dir/"bm25_index")
        chunk_metadata = [chunk.metadata for chunk in self.chunks]
        with open(self.processed_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunk_metadata, f, indent=4)

    def retrieve(self, query, k=5):
        query_tokens = bm25s.tokenize(query)
        docs, scores = self.retriever.retrieve(
            query_tokens,
            k=min(k, len(self.chunks))
        )

        retrieved_chunks = []

        for i, idx in enumerate(docs[0]):
            idx = int(idx)

            retrieved_chunks.append(
                {"document": self.chunks[idx], "score": float(scores[0, i])}
            )

        return retrieved_chunks


    def load_index(self):
        """Loads the saved BM25 index and chunk metadata from disk."""
        self.retriever = bm25s.BM25.load(
            self.processed_dir / "bm25_index", load_vocab=True
        )

        with open(self.processed_dir / "chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def retrieve(self, query, k=5):
        """Searches the index. Loads from disk automatically if not in memory."""
        if not hasattr(self, "chunks") or self.chunks is None:
            self.load_index()

        query_tokens = bm25s.tokenize(query)
        docs, scores = self.retriever.retrieve(query_tokens, k=min(k, len(self.chunks)))

        retrieved_chunks = []
        for i, idx in enumerate(docs[0]):
            idx = int(idx)
            chunk = self.chunks[idx]
            retrieved_chunks.append({"document": chunk, "score": float(scores[0, i])})

        return retrieved_chunks
