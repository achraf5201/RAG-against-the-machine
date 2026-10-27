"""Retrieval-augmented generation pipeline: index, search, answer."""

import bm25s
import json
import uuid
from tqdm import tqdm
from typing import Any, Dict, List, Union
from pathlib import Path
from src.chunking import Chunking
from src.own_recall import Recall
from src.models import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)
from src.answer import QwenChat


class Rag:
    """BM25-based retrieval-augmented generation over local documents.

    Raw files are read from ``data/raw``, the index is stored in
    ``data/processed`` and results are written to ``data/output``. These
    directories are created if missing.

    Attributes:
        retriever: The BM25 retriever.
        dir_path: Directory holding the saved index and chunk metadata.
        raw_dir: Directory holding the files to index.
        output: Directory where search and answer results are written.
        flag: When truthy, :meth:`search` writes its result to
            ``search.json``. :meth:`search_dataset` sets it to ``0``.
    """

    def __init__(self) -> None:
        """Create the retriever and make sure the data directories exist."""
        self.retriever = bm25s.BM25()
        self.dir_path = Path("data/processed")
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.raw_dir = Path("data/raw")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.output = Path("data/output")
        self.output.mkdir(parents=True, exist_ok=True)
        self.flag = 1

    @staticmethod
    def print_and_exit(e: BaseException) -> None:
        """Print an error and terminate the process.

        Args:
            e: The exception to print.

        Raises:
            SystemExit: Always.
        """
        print(e)
        exit()

    def index(self, max_chunk_size: int = 2000) -> None:
        """Chunk the raw files and build and save the BM25 index.

        Markdown, Python and text files under ``data/raw`` are chunked and
        indexed. The index is saved to ``data/processed/bm25_index`` and the
        chunk metadata to ``data/processed/chunks.json``.

        Args:
            max_chunk_size: Maximum size of a chunk, in characters. Defaults
                to 2000.

        Raises:
            SystemExit: If no text chunks were found.
        """
        self.chunking = Chunking(Path(self.raw_dir), max_chunk_size)
        self.chunking.chunking_markdown()
        self.chunking.chunking_python()
        self.chunking.chunking_text()
        self.chunks: List[Any] = self.chunking.chunks
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
        """Load the saved BM25 index and chunk metadata.

        Sets ``retriever`` and ``chunks``. Every chunk is validated against
        ``MinimalSource``.

        Raises:
            SystemExit: If the chunk metadata cannot be read or is invalid.
        """
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

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Retrieve the chunks most relevant to a query.

        The saved index is loaded first. If ``flag`` is truthy, the result is
        also written to ``data/output/search.json``.

        Args:
            query: The search query.
            k: Number of chunks to retrieve. Capped at the number of indexed
                chunks.

        Returns:
            Chunk metadata dictionaries (``file_path``,
            ``first_character_index``, ``last_character_index``), best match
            first.

        Raises:
            SystemExit: If the index cannot be loaded.
        """
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
        """Run :meth:`search` for every question of a dataset.

        The results are saved to
        ``<save_directory>/<dataset file name>_search_results.json``.

        Args:
            dataset_path: Path to a JSON file with a ``rag_questions`` list.
            k: Number of chunks to retrieve per question.
            save_directory: Directory where the results file is written. It is
                created if missing.

        Returns:
            The ``StudentSearchResults`` data as a dictionary.

        Raises:
            SystemExit: If the dataset cannot be read.
        """
        self.flag = 0
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.print_and_exit(e)

        search_results: List[Any] = []
        for d in tqdm(data["rag_questions"]):
            search_results.append(
                {
                    "question_id": d["question_id"],
                    "question": d["question"],
                    "retrieved_sources": self.search(d["question"], k),
                }
            )

        result: Union[StudentSearchResults, Dict[str, Any]]
        result = StudentSearchResults(search_results=search_results, k=k)
        result = result.model_dump()
        dataset_filename = Path(dataset_path).stem
        save_path = path / f"{dataset_filename}_search_results.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        return result

    def _load_chunk_text(self, chunk: Dict[str, Any]) -> str:
        """Read the text of a chunk from its source file.

        Args:
            chunk: Chunk metadata with ``file_path``, ``first_character_index``
                and ``last_character_index``.

        Returns:
            The characters of the file between the first (inclusive) and last
            (exclusive) indices. Undecodable bytes are replaced.

        Raises:
            OSError: If the file cannot be read.
        """
        text = Path(
            chunk["file_path"]).read_text(encoding="utf-8", errors="replace")
        return text[
            chunk["first_character_index"]: chunk["last_character_index"]]

    def answer(
            self, query: str, k: int) -> Union[Dict[str, Any], str]:
        """Answer a question using the retrieved chunks as context.

        The result is also written to ``data/output/answer.json``.

        Args:
            query: The question to answer.
            k: Number of chunks to retrieve as context.

        Returns:
            The ``StudentSearchResultsAndAnswer`` data as a dictionary, or the
            string ``"No context could be loaded for this question."`` if none
            of the retrieved chunks could be read.
        """
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
        result: Union[str, Dict[str, Any]]
        result = qwen.generate(context, query)
        final_answer: Union[Dict[str, Any], MinimalAnswer]
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
    ) -> Union[Dict[str, Any], str]:
        """Answer every question of a saved search-results file.

        The result is saved to ``<save_directory>/answer_dataset.json``. The
        reported ``k`` is always 10.

        Args:
            student_search_results_path: Path to a JSON file in
                ``StudentSearchResults`` format, as produced by
                :meth:`search_dataset`.
            save_directory: Directory where the answers file is written. It is
                created if missing.

        Returns:
            The ``StudentSearchResultsAndAnswer`` data as a dictionary, or the
            string ``"No context could be loaded for this question."`` as soon
            as a question has no readable chunk (nothing is saved then).

        Raises:
            SystemExit: If the file cannot be read or fails validation.
        """
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
            result: Union[str, Dict[str, Any]]
            result = qwen.generate(context, search["question"])
            final_answer: Dict[str, Any] = {
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
        """Print recall@k of saved search results against a reference dataset.

        Args:
            student_search_results_path: Path to the search-results JSON file.
            dataset_path: Path to the reference dataset JSON file.
        """
        obj = Recall()
        obj.recall_at_k(student_search_results_path, dataset_path)
