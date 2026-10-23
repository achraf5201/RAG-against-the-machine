import json


class Recall:
    def __init__(self):
        pass

    def IoU(self, a1, a2, b1, b2):
        intersection = max(0, min(a2, b2) - max(a1, b1))
        union = (a2 - a1) + (b2 - b1) - intersection
        return intersection / union if union > 0 else 0.0

    def open_and_read(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(e)
            exit()

    def recall_at_k(self, path, dataset_path):
        search = self.open_and_read(path)
        dataset = self.open_and_read(dataset_path)
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

        return scores
