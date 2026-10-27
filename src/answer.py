"""Answer generation with a local Qwen chat model."""

from typing import Any, cast
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenChat:
    """Wrapper around a Hugging Face Qwen3 chat model.

    Attributes:
        model_name: Hugging Face model identifier or local path.
        tokenizer: Tokenizer matching the model.
        model: The loaded causal language model.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Load the tokenizer and the model.

        Args:
            model_name: Hugging Face model identifier or local path. Defaults
                to ``"Qwen/Qwen3-0.6B"``. Weights are downloaded on first use.
        """
        self.model_name = model_name

        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, context: str, query: str) -> str:
        """Generate an answer to a question using only the given context.

        The model is instructed to answer concisely from the context and to
        say it does not know if the context is insufficient. Thinking mode is
        enabled and up to 32768 new tokens are generated; everything up to and
        including the last ``</think>`` token (id 151668) is discarded.

        Args:
            context: Text passages the answer must be based on.
            query: The question to answer.

        Returns:
            The generated answer without the thinking section and without
            leading or trailing newlines. If no ``</think>`` token is found,
            the whole generated text is returned.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the context. "
                    "Be concise. If the context is insufficient, "
                    "say you don't know."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(
            self.model.device)
        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=32768)

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = cast(str, self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        )).strip("\n")

        return content
