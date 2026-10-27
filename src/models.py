"""Pydantic models for RAG datasets, search results and answers."""

from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """A span of characters inside a source file.

    Attributes:
        file_path: Path of the file.
        first_character_index: Index of the first character of the span.
        last_character_index: Index just past the last character of the
            span (exclusive).
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without a reference answer.

    Attributes:
        question_id: Unique identifier. A UUID4 string by default.
        question: The question text.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its reference sources and reference answer.

    Attributes:
        sources: Spans of the source files that contain the answer.
        answer: The reference answer.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A collection of questions used to evaluate the RAG pipeline.

    Attributes:
        rag_questions: The questions, answered or not.
    """

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The sources retrieved for a single question.

    Attributes:
        question_id: Identifier of the question.
        question: The question text.
        retrieved_sources: Retrieved spans, best match first.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """The retrieved sources and the generated answer for a question.

    Attributes:
        answer: The generated answer.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Search results for a whole dataset.

    Attributes:
        search_results: One entry per question.
        k: Number of sources requested per question.
    """

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results and generated answers for a whole dataset.

    Attributes:
        search_results: One entry per question, including its answer.
        k: Number of sources requested per question.
    """

    search_results: List[MinimalAnswer]
    k: int
