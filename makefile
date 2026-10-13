index:
	uv run python -m src index -max_chunk_size 2000

search:
	uv run -m src search "How does continuous batching work?" 5

clean:
	rm -rf data