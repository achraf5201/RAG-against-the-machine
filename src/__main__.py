from src.chunking import Chunking
from src.bm25 import Rag


def open_and_write(filename, chunks):
    with open(filename, "w") as f:
        for chunk in chunks:
            # print(str(chunk.metadata))
            f.write(chunk.page_content)
            f.write("\n")
            f.write(str(chunk.metadata["source"]))
            f.write("\n\n\n")


def main():
    chunking = Chunking()
    chunking.chunking_markdown()
    chunking.chunking_python()

    open_and_write("file.txt", chunking.chunks)
    # print(chunking.chunks)
    rag = Rag(chunking.chunks)
    rag.indexing()
    print(rag.retrieve("Benchmarking vLLM"))


if __name__ == "__main__":
    main()
