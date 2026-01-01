bindings:
	cd ./scripts/modeldb/bindings && \
	python3 setup.py install && \
	cd ../../../../

e2e:
	CUDA_VISIBLE_DEVICES=0 python3 -m scripts.modeldb.main_pq \
	-f llama-3.1-8b-instruct.json \
	--dataset _synthetic \
	-M_key 32 \
	-M_value 32 \
	--nbits_key 8 \
	--nbits_value 8 \
	--sink_tokens 0 \
	-m \
	--half \
	-p baseline evaluation 

breakdown:
	CUDA_VISIBLE_DEVICES=0 python3 -m scripts.modeldb.main_pq \
	-f llama-3.1-8b-instruct.json \
	--dataset _synthetic \
	-M_key 32 \
	-M_value 32 \
	--nbits_key 8 \
	--nbits_value 8 \
	--sink_tokens 0 \
	-m \
	--half \
	--breakdown \
	-p baseline evaluation 


training:
	@datasets="qasper"; \
	for dataset in $$datasets; do \
		echo "Running dataset: $$dataset"; \
		CUDA_VISIBLE_DEVICES=0 python3 -m scripts.modeldb.main_pq \
		--dataset $$dataset  \
		--cluster_dataset qasper \
		-f llama-3.1-8b-instruct.json \
		-M_key 32 \
		-M_value 32 \
		--nbits_key 8 \
		--nbits_value 8 \
		-m \
		--half \
		-p sampling training || echo "$$dataset failed"; \
	done


longbench:
	@datasets="qasper multifieldqa_en hotpotqa 2wikimqa trec samsum triviaqa passage_count passage_retrieval_en lcc repobench-p gov_report multi_news"; \
	for dataset in $$datasets; do \
		echo "Running dataset: $$dataset"; \
		CUDA_VISIBLE_DEVICES=0 python3 -m scripts.modeldb.main_pq \
		--dataset $$dataset  \
		--cluster_dataset qasper \
		-f llama-3.1-8b-instruct.json \
		-M_key 32 \
		-M_value 32 \
		--nbits_key 8 \
		--nbits_value 8 \
		--sink_tokens 16 \
		-m \
		--half \
		-p evaluation || echo "$$dataset failed"; \
	done


math_baseline:
	CUDA_VISIBLE_DEVICES=0 python3 -m scripts.modeldb.main_pq \
	-f deepseek-r1-distill-qwen-14b.json \
	--dataset math \
	-M_key 32 \
	-M_value 32 \
	--nbits_key 12 \
	--nbits_value 12 \
	--sink_tokens 0 \
	-m \
	--half \
	--save_path ./math_output/math/DeepSeek-R1-Distill-Qwen-14B-baseline.jsonl \
	-p baseline



math_evaluation:
	@seeds="42 43"; \
	for seed in $$seeds; do \
		echo "Running with seed=$$seed..."; \
		CUDA_VISIBLE_DEVICES=0 python3 -m scripts.modeldb.main_pq \
			--dataset math \
			--cluster_dataset qasper \
			-f deepseek-r1-distill-qwen-14b.json \
			-M_key 32 \
			-M_value 32 \
			--nbits_key 12 \
			--nbits_value 12 \
			--sink_tokens 0 \
			-m \
			--half \
			--seed $$seed \
			--save_path ./math_output/math/DeepSeek-R1-Distill-Qwen-14B-32-32-12-12-$$seed.jsonl \
			-p evaluation; \
	done

