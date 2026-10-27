"""Command-line entry point for the RAG pipeline."""

from src.rag import Rag
import fire


def main() -> None:
    """Expose the public methods of ``Rag`` as a command-line interface.

    Uses ``python-fire`` so that ``index``, ``search``, ``search_dataset``,
    ``answer``, ``answer_dataset`` and ``evaluate`` become sub-commands.
    """
    fire.Fire(Rag)


if __name__ == "__main__":
    main()
