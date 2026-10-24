from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenChat:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, context, query):
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

        content = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip("\n")

        return content
