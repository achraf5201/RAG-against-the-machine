index:
	uv run python -m src index -max_chunk_size 2000

search:
	uv run -m src search "How to configure OpenAI server?" 5

search_dataset:
	uv run -m src search_dataset -dataset_path "datasets_public/public/UnansweredQuestions/dataset_code_public.json" -k 5 -save_directory "data/output/search_results/"

answer:
	uv run -m src answer "How to configure OpenAI server?" 5

answer_dataset:
	ls

clean:
	rm -rf data