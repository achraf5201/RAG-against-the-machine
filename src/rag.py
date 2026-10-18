import bm25s
import json
import fire
import uuid
import torch
from pathlib import Path
from src.chunking import Chunking
from src.models import MinimalSearchResults, StudentSearchResults
from src.answer import QwenChat
from transformers import AutoModelForCausalLM, AutoTokenizer


class Rag:
    def __init__(self):
        self.retriever = bm25s.BM25()
        self.processed_dir = Path("data/processed")
        self.raw_dir = Path("data/raw")

    def index(self, max_chunk_size=2000):
        self.chunking = Chunking(Path(self.raw_dir), max_chunk_size)
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

    def search_dataset(self, dataset_path, k, save_directory):
        data = {}

        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        search_results = []

        for d in data["rag_questions"]:
            search_results.append(
                {
                    "question_id": d["question_id"],
                    "question": d["question"],
                    "retrieved_sources": self.search(d["question"], k),
                }
            )

        result = StudentSearchResults(search_results=search_results, k=k)
        result = result.model_dump()
        with open(save_directory + "search.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        return result

    def _load_chunk_text(self, chunk):
        text = Path(chunk["file_path"]).read_text(encoding="utf-8", errors="replace")
        return text[chunk["first_character_index"] : chunk["last_character_index"]]

    def answer(self, query, k=5, max_new_tokens=256):
        retrieved_chunks = self.search(query, k)

        self.contents = []
        for chunk_info in retrieved_chunks:
            try:
                self.contents.append(self._load_chunk_text(chunk_info))
            except OSError as e:
                print(f"Could not read {chunk_info['file_path']}: {e}")

        if not self.contents:
            return "No context could be loaded for this question."

        context = "\n\n".join(
            f"[Source {i}: {c['file_path']}]\n{text}"
            for i, (c, text) in enumerate(zip(retrieved_chunks, self.contents), 1)
        )
        qwen = QwenChat()
        answer = qwen.generate(context, query)
        final_answer = {
            "sources": [chunk_info for chunk_info in retrieved_chunks],
            "answer": answer
        }
        return final_answer


# """
# To configure the OpenAI server, set the API key and API base URL as specified in the examples. For example:  - **API Key**: Use `"EMPTY"` (as shown in the context). - **API Base**: Use `"http://localhost:8000/v1"`.  These parameters are required to connect to the vLLM server using the OpenAI-compatible Completions API.
# The configuration of the OpenAI server can be done by modifying the `openai_api_key` and `openai_api_base` variables in the script.
# """
