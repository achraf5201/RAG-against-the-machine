import bm25s
import json
import fire
import uuid
from pathlib import Path
from src.chunking import Chunking
from src.models import MinimalSearchResults
from transformers import AutoModelForCausalLM, AutoTokenizer    


class Rag:
    def __init__(self):
        self.retriever = bm25s.BM25()
        self.processed_dir = Path("data/processed")
        self.raw_dir = Path("data/raw")

    def index(self, max_chunk_size=2000):
        self.chunking = Chunking(Path("vllm-0.10.1"), max_chunk_size)
        self.chunking.chunking_markdown()
        self.chunking.chunking_python()
        self.chunking.chunking_text()
        self.chunks = self.chunking.chunks
        self.texts = [chunk.page_content for chunk in self.chunks]

        self.corpus_tokens = bm25s.tokenize(self.texts)
        self.retriever.index(self.corpus_tokens)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.retriever.save(self.processed_dir / "bm25_index")
        chunk_metadata = [chunk.metadata for chunk in self.chunks]
        with open(self.processed_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunk_metadata, f, indent=4)

    def load_index(self):
        self.retriever = bm25s.BM25.load(
            self.processed_dir / "bm25_index", load_vocab=True
        )

        with open(self.processed_dir / "chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def search(self, query, k=5):
        try:
            self.load_index()
        except Exception as e:
            print(e)
            exit()

        query_tokens = bm25s.tokenize(query)
        result, scores = self.retriever.retrieve(
            query_tokens, k=min(k, len(self.chunks))
        )
        retrieved_chunks = []

        for idx in result[0]:
            idx = int(idx)
            chunk = self.chunks[idx]
            retrieved_chunks.append(chunk)

        obj = MinimalSearchResults(
            question_id=str(uuid.uuid4()),
            question=query,
            retrieved_sources=retrieved_chunks,
        )
        data = obj.model_dump()
        path = Path("data/output")
        path.mkdir(parents=True, exist_ok=True)
        with open("data/output/result.json", "w") as f:
            json.dump(data, f, indent=4)

        return retrieved_chunks

    def _load_content(self, filename):
        with open(filename, "r") as f:
            self.contents.append(f.read())

    def answer(self, query, k=5):
        retrieved_chunks = self.search(query, k)
        self.contents = []
        for chunk_info in retrieved_chunks:
            with open(chunk_info["file_path"]) as f:
                f.seek(chunk_info["first_character_index"])
                self.contents.append(
                    f.read(
                        chunk_info["last_character_index"]
                        - chunk_info["first_character_index"]
                    )
                )
        for f in self.contents:
            print(f)
            print("-" * 50)
        # ai answer
