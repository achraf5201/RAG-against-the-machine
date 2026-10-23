index:
	uv run -m src index -max_chunk_size 2000

search:
	uv run -m src search "What activation formats does the fused batched MoE layer return in vLLM?" 5

search_dataset:
	uv run -m src search_dataset -dataset_path "data/datasets_public/public/UnansweredQuestions/dataset_docs_public.json" -k 10 -save_directory "data/output/search_results/"

answer:
	uv run -m src answer "What activation formats does the fused batched MoE layer return in vLLM?" 5

answer_dataset:
	uv run -m src answer_dataset -student_search_results_path "data/output/search_results/search.json" -save_directory "data/output/answer_results/"

evaluate:
	uv run -m src evaluate -student_search_results_path data/output/search_results/search.json -dataset_path data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json

mkdir:
	mkdir -p data/raw

unzip: 
	unzip vllm-0.10.1.zip -d data/raw/
	unzip datasets_public.zip -d data

clean:
	rm -rf data

flake8:
	uv run -m flake8 src