from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)
from pathlib import Path
from models import MinimalSource


class Chunking:
    def __init__(self, max_chunk_size=200):
        self.chunks = []
        self.path = Path()
        self.max_chunk_size = max_chunk_size
        self.text_splitter_py = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.max_chunk_size,
            chunk_overlap=50,
            add_start_index=True,
        )
        self.text_splitter_md = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=self.max_chunk_size,
            chunk_overlap=50,
            add_start_index=True,
        )

    def chunk_docs(self):
        all_files = (
            list(self.path.rglob("*.py"))
            + list(self.path.rglob("*.md"))
            + list(self.path.rglob("*.txt"))
        )
        master_chunks = []

        for file in all_files:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()

            if file.suffix == ".py":
                doc = self.text_splitter_py.create_documents(
                    texts=[content], metadatas=[{"file_path": str(file)}]
                )

            elif file.suffix in [".md", ".txt"]:
                doc = self.text_splitter_md.create_documents(
                    texts=[content], metadata=[{"file_path": str(file)}]
                )

            master_chunks.extend(doc)
        return master_chunks

    def getting_min_src(self):
        master_chunks = self.chunk_docs()
        min_lst = []
        raw_lst = []

        for chunk in master_chunks:
            start_index = chunk.metadata["start_index"]
            last_index = start_index + len(chunk.page_content)
            min_obj = MinimalSource(
                file_path=chunk.metadata["file_path"],
                first_character_index=start_index,
                last_character_index=last_index,
            )
            min_lst.append(min_obj)
            raw_lst.append(chunk.page_content)
        return min_lst, raw_lst


chnk = Chunking()
print(chnk.getting_min_src())
