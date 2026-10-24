import os
import bm25s
import json
import uuid
from tqdm import tqdm
from typing import Dict, List
from pathlib import Path
from src.chunking import Chunking
from src.own_recall import Recall
from src.models import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
    RagDataset
)
from src.answer import QwenChat
from transformers import AutoModelForCausalLM, AutoTokenizer


class Rag:
    def __init__(self) -> None:
        self.retriever = bm25s.BM25()
        self.dir_path = Path("data/processed")
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.raw_dir = Path("data/raw")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.output = Path("data/output")
        self.output.mkdir(parents=True, exist_ok=True)
        self.flag = 1

    @staticmethod
    def print_and_exit(e) -> None:
        print(e)
        exit()

    def index(self, max_chunk_size: int = 2000) -> None:
        self.chunking = Chunking(Path(self.raw_dir), max_chunk_size)
        self.chunking.chunking_markdown()
        self.chunking.chunking_python()
        self.chunking.chunking_text()
        self.chunks = self.chunking.chunks
        self.texts = [chunk.page_content for chunk in self.chunks]
        try:
            if len(self.texts) == 0:
                raise ValueError("No text chunks were found.")
        except ValueError as e:
            self.print_and_exit(e)

        self.corpus_tokens = bm25s.tokenize(self.texts)
        self.retriever.index(self.corpus_tokens)
        
        self.retriever.save(self.dir_path / "bm25_index")
        chunk_metadata = [chunk.metadata for chunk in self.chunks]
        with open(self.dir_path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunk_metadata, f, indent=4)

    def _load_index(self) -> None:
        self.retriever = bm25s.BM25.load(
            self.dir_path / "bm25_index", load_vocab=True)
        try:
            with open(
                self.dir_path / "chunks.json", "r", encoding="utf-8"
            ) as f:
                self.chunks = json.load(f)

                for chunk in self.chunks:
                    MinimalSource(**chunk)

        except BaseException as e:
            self.print_and_exit(e)

    def search(self, query: str, k: int) -> Dict:
        try:
            self._load_index()
        except Exception as e:
            self.print_and_exit(e)

        query_tokens = bm25s.tokenize(query)
        result, scores = self.retriever.retrieve(
            query_tokens, k=min(k, len(self.chunks))
        )
        retrieved_chunks = []

        for idx in result[0]:
            idx = int(idx)
            chunk = self.chunks[idx]
            retrieved_chunks.append(chunk)

        result = MinimalSearchResults(
            question_id=str(uuid.uuid4()),
            question=query,
            retrieved_sources=retrieved_chunks,
        ).model_dump()

        if self.flag:
            with open(self.output / "search.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
        return retrieved_chunks

    def search_dataset(
            self, dataset_path: str, k: int, save_directory: str) -> dict:
        self.flag = 0
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.print_and_exit(e)

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
        dataset_filename = Path(dataset_path).stem
        save_path = path / f"{dataset_filename}_search_results.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        return result

    def _load_chunk_text(self, chunk: str) -> str:
        text = Path(
            chunk["file_path"]).read_text(encoding="utf-8", errors="replace")
        return text[
            chunk["first_character_index"]: chunk["last_character_index"]]

    def answer(self, query: str, k: int) -> Dict:
        retrieved_chunks = self.search(query, k)

        contents = []
        for chunk_info in retrieved_chunks:
            try:
                contents.append(self._load_chunk_text(chunk_info))
            except OSError as e:
                print(f"Could not read {chunk_info['file_path']}: {e}")

        if not contents:
            return "No context could be loaded for this question."

        context = "\n\n".join(contents)
        qwen = QwenChat()
        sources = [
            MinimalSource(**chunk_info) for chunk_info in retrieved_chunks]
        result = qwen.generate(context, query)
        final_answer = {
            "question_id": str(uuid.uuid4()),
            "question": query,
            "retrieved_sources": sources,
            "answer": result,
        }
        final_answer = MinimalAnswer(**final_answer)
        result = StudentSearchResultsAndAnswer(
            search_results=[final_answer], k=k).model_dump()
        with open(
            self.output / "answer.json", "w", encoding="utf-8"
        ) as f:
            json.dump(result, f, indent=4)
        return result

    def answer_dataset(
            self, student_search_results_path: str, save_directory: str
    ) -> Dict:
        answer_dataset_result = []

        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                StudentSearchResults(**data)

        except Exception as e:
            self.print_and_exit(e)

        for search in data["search_results"]:
            contents = []
            for chunk_info in search["retrieved_sources"]:
                try:
                    contents.append(self._load_chunk_text(chunk_info))
                except OSError as e:
                    print(f"Could not read {chunk_info['file_path']}: {e}")
            if not contents:
                return "No context could be loaded for this question."

            context = "\n\n".join(contents)
            qwen = QwenChat()
            sources = [
                MinimalSource(**chunk_info)
                for chunk_info in search["retrieved_sources"]
            ]
            result = qwen.generate(context, search["question"])
            final_answer = {
                "question_id": search["question_id"],
                "question": search["question"],
                "retrieved_sources": sources,
                "answer": result,
            }
            answer_dataset_result.append(
                MinimalAnswer(**final_answer))
        result = StudentSearchResultsAndAnswer(
            search_results=answer_dataset_result, k=10).model_dump()
        with open(
            path / "answer_dataset.json", "w", encoding="utf-8"
        ) as f:
            json.dump(result, f, indent=4)
        return result

    def evaluate(
            self, student_search_results_path: str, dataset_path: str) -> None:
        obj = Recall()
        obj.recall_at_k(student_search_results_path, dataset_path)
