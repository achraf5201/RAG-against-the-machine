from src.chunking import Chunking
from src.rag import Rag
from pathlib import Path
import json
import fire


def main():
    fire.Fire(Rag)


if __name__ == "__main__":
    main()
