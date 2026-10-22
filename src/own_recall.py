import json


class recall:
    def __init__(self):
        pass

    def IoU(self):
        pass

    def open_and_read(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(e)
            exit()

    def recall_at_k(self, path, dataset_path):
        search = self.open_and_read(path)
        ls = []
        dataset = self.open_and_read(dataset_path)
        for i in range(len(dataset["rag_questions"])):
            if dataset["rag_questions"][i]["question_id"] == search["search_results"][i]["question_id"]:
                for retrieved_sources in search["search_results"][i]["retrieved_sources"]:
                    if any(
                        source["file_path"] == retrieved_sources["file_path"]
                        for source in dataset["rag_questions"][i]["sources"]
                    ):
                        ls.append("good")
                        break
        print(len(ls))
