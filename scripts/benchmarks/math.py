import json
import random
import argparse

import numpy as np
from tqdm import tqdm

import torch


dataset2key = {
    "gsm8k": ["question", "answer"],
    "aime24": ["question", "answer"],
    "math": ["problem", "answer"],
    "amc": ["problem", "answer"],
}

dataset2max_length = {
    "gsm8k": 16384,
    "aime24": 32768,
    "math": 16384,
    "amc": 16384,
}

prompt_template = "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"

def math(model, tokenizer, *, dataset, datasets_root, save_path, device, stride=None, verbose=True, cache_clear_func=None, **kwargs):
    if model.config.model_type == "qwen3":
        print(model.config.model_type)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        dataset_path = f"{datasets_root}/{dataset}.jsonl"
        
        fout = open(save_path, "w")

        prompts = []
        test_data = []

        with open(dataset_path) as f:
            for index, line in enumerate(f):
                example = json.loads(line)
                question_key = dataset2key[dataset][0]

                question = example[question_key]
                example["question"] = question
                # prompt = prompt_template.format(**example)
                prompt_content = f"You are given a math problem.\n\nProblem: {question}\n\nYou need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\nProvide the final answer in the format: Final answer: \\boxed{{}}"
            
                messages = [{"role": "user", "content": prompt_content}]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True  # Enable thinking mode
                )

                example["prompt"] = prompt
                example["index"] = index
                prompts.append(prompt)
                test_data.append(example)


        for i in tqdm(range(0, len(prompts), 1)):
            batch_prompts = prompts[i : i + 1]
            tokenized_prompts = tokenizer(
                batch_prompts,
                return_tensors="pt",
            ).to("cuda")

            prefill_lengths = tokenized_prompts["attention_mask"].sum(dim=1).tolist()

            if cache_clear_func is not None:
                cache_clear_func()
                
            output = model.generate(
                **tokenized_prompts,
                max_length=dataset2max_length[dataset],
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
            )

            batch_token_stats = []
            for j in range(output.size(0)):
                total_tokens = int((output[j] != tokenizer.pad_token_id).sum().item())

                prefill = prefill_lengths[j]
                output_tokens = total_tokens - prefill

                batch_token_stats.append(
                    {
                        "sample_idx": i + j,
                        "prefill_tokens": prefill,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                    }
                )

            batch_outputs = tokenizer.batch_decode(
                [output[j][prefill_lengths[j] :] for j in range(output.size(0))],
                skip_special_tokens=True,
            )

            torch.cuda.empty_cache()
            

            for j in range(len(batch_outputs)):
                sample_idx = batch_token_stats[j]["sample_idx"]
                test_data[sample_idx]["prompt"] = batch_prompts[j]
                test_data[sample_idx]["output"] = batch_outputs[j]
                test_data[sample_idx]["prefill_tokens"] = batch_token_stats[j]["prefill_tokens"]
                test_data[sample_idx]["output_tokens"] = batch_token_stats[j]["output_tokens"]
                test_data[sample_idx]["total_tokens"] = batch_token_stats[j]["total_tokens"]
                test_data[sample_idx]["sample_idx"] = batch_token_stats[j]["sample_idx"]

                fout.write(json.dumps(test_data[sample_idx], ensure_ascii=False) + "\n")

        fout.close()

    else:
        print(model.config.model_type)
        dataset_path = f"{datasets_root}/{dataset}.jsonl"
        
        fout = open(save_path, "w")

        prompts = []
        test_data = []

        with open(dataset_path) as f:
            for index, line in enumerate(f):
                example = json.loads(line)
                question_key = dataset2key[dataset][0]

                question = example[question_key]
                example["question"] = question
                prompt = prompt_template.format(**example)

                example["prompt"] = prompt
                example["index"] = index
                prompts.append(prompt)
                test_data.append(example)


        for i in tqdm(range(0, len(prompts), 1)):
            batch_prompts = prompts[i : i + 1]
            tokenized_prompts = tokenizer(
                batch_prompts,
                padding="longest",
                return_tensors="pt",
                add_special_tokens=True,
            ).to("cuda")

            prefill_lengths = tokenized_prompts["attention_mask"].sum(dim=1).tolist()

            if cache_clear_func is not None:
                cache_clear_func()
                
            output = model.generate(
                **tokenized_prompts,
                max_length=dataset2max_length[dataset],
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
            )

            batch_token_stats = []
            for j in range(output.size(0)):
                total_tokens = int((output[j] != tokenizer.pad_token_id).sum().item())

                prefill = prefill_lengths[j]
                output_tokens = total_tokens - prefill

                batch_token_stats.append(
                    {
                        "sample_idx": i + j,
                        "prefill_tokens": prefill,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                    }
                )

            batch_outputs = tokenizer.batch_decode(
                [output[j][prefill_lengths[j] :] for j in range(output.size(0))],
                skip_special_tokens=True,
            )

            torch.cuda.empty_cache()
            

            for j in range(len(batch_outputs)):
                sample_idx = batch_token_stats[j]["sample_idx"]
                test_data[sample_idx]["prompt"] = batch_prompts[j]
                test_data[sample_idx]["output"] = batch_outputs[j]
                test_data[sample_idx]["prefill_tokens"] = batch_token_stats[j]["prefill_tokens"]
                test_data[sample_idx]["output_tokens"] = batch_token_stats[j]["output_tokens"]
                test_data[sample_idx]["total_tokens"] = batch_token_stats[j]["total_tokens"]
                test_data[sample_idx]["sample_idx"] = batch_token_stats[j]["sample_idx"]

                fout.write(json.dumps(test_data[sample_idx], ensure_ascii=False) + "\n")

        fout.close()

