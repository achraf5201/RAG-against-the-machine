"""Split source documents into overlapping, character-indexed chunks."""

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)
from pathlib import Path
from typing import List, Optional, cast
from src.models import MinimalSource


class Chunking:
    """Load files from a directory and split them into chunks.

    Markdown, Python and plain-text files are supported. Every chunk keeps
    ``file_path``, ``first_character_index`` and ``last_character_index``
    in its metadata so that it can be located in the original file.

    Attributes:
        chunks: Chunks produced so far.
        path: Directory searched recursively for files.
        max_chunk_size: Maximum chunk size, in characters.
    """

    def __init__(self, path: Path, max_chunk_size: int = 200) -> None:
        """Initialize the chunker.

        Args:
            path: Directory searched recursively for files to chunk.
            max_chunk_size: Maximum size of a chunk, in characters. Defaults to
                200.
        """
        self.chunks: List[Document] = []
        self.path = path
        self.max_chunk_size = max_chunk_size

    def load_documents(self, extension: str) -> List[Document]:
        """Read every file matching a glob pattern under ``path``.

        Args:
            extension: Glob pattern to match, e.g. ``"*.py"``.

        Returns:
            One ``Document`` per file, holding the full UTF-8 content and the
            file path in its ``file_path`` metadata.
        """
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

    def chunk(
            self, language: Optional[Language], extension: str) -> None:
        """Split the files matching a pattern into chunks.

        Chunks overlap by 100 characters. For ``"*.py"`` and ``"*.md"`` the
        splitter uses language-aware separators; for ``"*.txt"`` it uses the
        default separators. The resulting chunks are appended to ``chunks``,
        with ``start_index`` replaced by ``first_character_index`` and
        ``last_character_index`` (exclusive end) added to their metadata. Each
        chunk's location is validated against ``MinimalSource``.

        Args:
            language: Language whose splitting rules are used for ``"*.py"``
                and ``"*.md"``. May be ``None`` for ``"*.txt"``.
            extension: Glob pattern selecting the files. Supported values are
                ``"*.py"``, ``"*.md"`` and ``"*.txt"``.

        Raises:
            pydantic.ValidationError: If a chunk's location metadata is
                invalid.
        """
        documents = self.load_documents(extension)

        if extension in ["*.py", "*.md"]:
            text_splitter = RecursiveCharacterTextSplitter.from_language(
                language=cast(Language, language),
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
        """Chunk all Markdown (``*.md``) files under ``path``."""
        self.chunk(
            language=Language.MARKDOWN,
            extension="*.md",
        )

    def chunking_python(self) -> None:
        """Chunk all Python (``*.py``) files under ``path``."""
        self.chunk(
            language=Language.PYTHON,
            extension="*.py",
        )

    def chunking_text(self) -> None:
        """Chunk all plain-text (``*.txt``) files under ``path``."""
        self.chunk(
            language=None,
            extension="*.txt",
        )
