"""Recall@k evaluation of search results against a reference dataset."""

import json
from src.models import RagDataset
from typing import Dict


class Recall:
    """Compute recall@k of retrieved sources against reference sources.

    Attributes:
        recall_res: Number of questions answered correctly at each cut-off
            (``recall@1``, ``recall@3``, ``recall@5``, ``recall@10``). Only
            set once :meth:`recall_at_k` has been called.
    """

    def __init__(self) -> None:
        """Initialize the evaluator (takes no arguments and keeps no state)."""
        pass

    def IoU(self, a1: int, a2: int, b1: int, b2: int) -> float:
        """Compute the intersection over union of two character ranges.

        Args:
            a1: Start index of the first range.
            a2: End index of the first range.
            b1: Start index of the second range.
            b2: End index of the second range.

        Returns:
            The length of the overlap divided by the length of the union of
            the two ranges, or ``0.0`` if the union is empty.
        """
        intersection = max(0, min(a2, b2) - max(a1, b1))
        union = (a2 - a1) + (b2 - b1) - intersection
        return intersection / union if union > 0 else 0.0

    def open_and_read(self, path: str) -> Dict:
        """Read and parse a JSON file.

        Args:
            path: Path of the JSON file to read.

        Returns:
            The parsed JSON content.

        Raises:
            SystemExit: If the file cannot be opened or parsed. The error is
                printed before the process exits.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data: Dict = json.load(f)
                return data
        except Exception as e:
            print(e)
            exit()

    def recall_at_k(self, path: str, dataset_path: str) -> Dict:
        """Compute recall@1, @3, @5 and @10 of search results for a dataset.

        Both files are validated against ``RagDataset``. Results and dataset
        questions are paired by position and must share the same
        ``question_id``. A question counts as a hit at rank ``j`` when the
        ``j``-th retrieved source comes from the same file as the first
        reference source and overlaps it with an IoU of at least ``0.05``;
        only the first matching rank of each question is counted. Scores are
        normalised by a fixed total of 100 questions. The counts and scores
        are also printed.

        Args:
            path: Path to the search-results JSON file
                (``StudentSearchResults`` format).
            dataset_path: Path to the reference dataset JSON file
                (``RagDataset`` format with answered questions).

        Returns:
            A mapping from ``"recall@1"``, ``"recall@3"``, ``"recall@5"`` and
            ``"recall@10"`` to the corresponding score.

        Raises:
            SystemExit: If either file cannot be read or fails validation.
        """
        try:
            search = self.open_and_read(path)
            RagDataset(rag_questions=search["search_results"])
            dataset = self.open_and_read(dataset_path)
            RagDataset(rag_questions=dataset["rag_questions"])
        except Exception as e:
            print(e)
            exit()
        rag_questions = dataset["rag_questions"]
        search_results = search["search_results"]
        self.recall_res = {
            "recall@1": 0,
            "recall@3": 0,
            "recall@5": 0,
            "recall@10": 0
        }

        print(len(dataset["rag_questions"]))
        for i in range(len(rag_questions)):
            if (
                rag_questions[i]["question_id"]
                == search_results[i]["question_id"]
            ):
                for j in range(len(search_results[i]["retrieved_sources"])):
                    if (
                        rag_questions[i]["sources"][0]["file_path"]
                        == search_results[i]["retrieved_sources"][j][
                            "file_path"]
                    ):
                        value = self.IoU(
                            rag_questions[i]["sources"][0][
                                "first_character_index"],
                            rag_questions[i]["sources"][0][
                                "last_character_index"],
                            search_results[i]["retrieved_sources"][j][
                                "first_character_index"
                            ],
                            search_results[i]["retrieved_sources"][j][
                                "last_character_index"
                            ],
                        )
                        if value >= 0.05:
                            if j + 1 <= 1:
                                self.recall_res["recall@1"] += 1
                            if j + 1 <= 3:
                                self.recall_res["recall@3"] += 1
                            if j + 1 <= 5:
                                self.recall_res["recall@5"] += 1
                            if j + 1 <= 10:
                                self.recall_res["recall@10"] += 1
                            break
        scores = {k: count / 100 for k, count in self.recall_res.items()}

        print(f"Total questions: {100}")

        for key in self.recall_res:
            print(
                f"{key}: {self.recall_res[key] / 100:.3f} "
                f"({float(self.recall_res[key])}%)"
            )
        print("Recall@K:", scores)
        print(type(scores))
        return scores
