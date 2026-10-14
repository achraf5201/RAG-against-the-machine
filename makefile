index:
	uv run python -m src index -max_chunk_size 2000

search:
	uv run -m src search "How to configure OpenAI server?" 5

answer:
	uv run -m src answer "How to configure OpenAI server?" 5

clean:
	rm -rf data