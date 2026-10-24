from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)
from pathlib import Path
from src.models import MinimalSource


class Chunking:
    def __init__(self, path: str, max_chunk_size: int = 200) -> None:
        self.chunks = []
        self.path = path
        self.max_chunk_size = max_chunk_size

    def load_documents(self, extension: str) -> Document:
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

    def chunk(self, language: Language, extension: str) -> None:
        documents = self.load_documents(extension)

        if extension in ["*.py", "*.md"]:
            text_splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=self.max_chunk_size,
                chunk_overlap=100,
                add_start_index=True,
            )
        elif extension == "*.txt":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.max_chunk_size,
                chunk_overlap=100,
                add_start_index=True,
            )

        chunks = text_splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata[
                "first_character_index"] = chunk.metadata["start_index"]
            chunk.metadata.pop("start_index", None)
            start = chunk.metadata["first_character_index"]

            chunk.metadata[
                "last_character_index"] = start + len(chunk.page_content)
            MinimalSource(
                file_path=chunk.metadata["file_path"],
                first_character_index=chunk.metadata["first_character_index"],
                last_character_index=chunk.metadata["last_character_index"],
            )
        self.chunks.extend(chunks)

    def chunking_markdown(self) -> None:
        self.chunk(
            language=Language.MARKDOWN,
            extension="*.md",
        )

    def chunking_python(self) -> None:
        self.chunk(
            language=Language.PYTHON,
            extension="*.py",
        )

    def chunking_text(self) -> None:
        self.chunk(
            language=None,
            extension="*.txt",
        )
