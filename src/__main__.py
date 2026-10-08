from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)


class Rag:
    def __init__(self):
        self.chunks = []
        self.path = Path("vllm-0.10.1")

    def load_documents(self, extension, file_type):
        documents = []

        for filename in self.path.rglob(extension):
            with open(filename, encoding="utf-8") as f:
                content = f.read()

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(filename),
                        "filename": filename.name,
                        "file_type": file_type,
                    },
                )
            )

        return documents

    def chunk(self, language, extension, file_type):
        documents = self.load_documents(extension, file_type)

        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=200,
            chunk_overlap=20,
            add_start_index=True,
        )

        chunks = text_splitter.split_documents(documents)

        for chunk in chunks:
            start = chunk.metadata["start_index"]

            chunk.metadata["end_index"] = (
                start + len(chunk.page_content)
            )

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


r = Rag()

r.chunking_markdown()
r.chunking_python()

for chunk in r.chunks:
    print(chunk.page_content)
    print(chunk.metadata)
    print("\n---\n")