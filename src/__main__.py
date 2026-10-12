from src.chunking import Chunking
from src.bm25 import Rag
from pathlib import Path
import json


def open_and_write(filename, chunks):
    with open(filename, "w", encoding="utf-8") as f:
        chunk_metadata = []
        for chunk in chunks:
            chunk_metadata.append(chunk.metadata)
        json.dump(chunk_metadata, f, indent=4)


def main():
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    chunking = Chunking(Path("vllm-0.10.1"))
    chunking.chunking_markdown()
    chunking.chunking_python()

    # open_and_write("data/processed/chunks.json", chunking.chunks)
    r = Rag(chunking.chunks)
    r.indexing()
    results = r.retrieve(
        query="How does continuous batching work?",
        k=5,
    )

    for rank, result in enumerate(results, start=1):
        document = result["document"]

        # print(f"\nRank: {rank}")
        # print(f"Score: {result['score']:.4f}")
        # print(f"Source: {document.metadata['source']}")
        # print(f"Metadata: {document.metadata}")
        # print(f"Text: {document.page_content[:300]}")
        # print("-" * 50)


if __name__ == "__main__":
    main()
