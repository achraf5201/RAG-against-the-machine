import json


class recall:
    def __init__(self):
        pass

    def IoU(self):
        pass

    def recall_at_k(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.search_dataset = json.load(f)
        print(self.search_dataset)
