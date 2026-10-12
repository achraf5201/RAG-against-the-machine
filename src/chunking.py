from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)
from pathlib import Path
import fire


class Chunking:
    def __init__(self, path):
        self.chunks = []
        self.path = path

    def load_documents(self, extension, file_type):
        documents = []

        for filename in self.path.rglob(extension):
            with open(filename, encoding="utf-8") as f:
                content = f.read()

            documents.append(
                Document(
                    page_content=content,
                    metadata={"file_path": str(filename)},
                )
            )

        return documents

    def chunk(self, language, extension, file_type):
        documents = self.load_documents(extension, file_type)

        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=1000,
            chunk_overlap=20,
            add_start_index=True,
        )

        chunks = text_splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata["first_character_index"] = chunk.metadata["start_index"]
            chunk.metadata.pop("start_index", None)
            start = chunk.metadata["first_character_index"]

            chunk.metadata["last_character_index"] = start + len(chunk.page_content)

        self.chunks.extend(chunks)

        return chunks

    def chunking_markdown(self):
        return self.chunk(
            language=Language.MARKDOWN,
            extension="*.md",
            file_type="markdown",
        )

    def chunking_python(self):
        return self.chunk(
            language=Language.PYTHON,
            extension="*.py",
            file_type="python",
        )
