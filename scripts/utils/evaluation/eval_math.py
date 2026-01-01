import os
import argparse
import json
import csv
from tqdm import tqdm

from evaluate import evaluate
from utils import save_jsonl
from python_executor import PythonExecutor
from data_loader import load_data_vanilla
from parser import choice_answer_clean, parse_ground_truth, run_execute


def parse_args():
    parser = argparse.ArgumentParser()
    # Experiment related parameters
    parser.add_argument("--exp_name", default="QwQ-32B-Preview", type=str)
    # Prompt type, such as cot, pal, etc.
    parser.add_argument("--prompt_type", default="cot", type=str)
    # Specify the folder containing JSON files to be evaluated (not a single JSON file)
    parser.add_argument(
        "--base_dir",
        default="./results",
        type=str,
        help="Folder containing JSON/JSONL files to be evaluated",
    )
    # Output directory
    parser.add_argument("--output_dir", default="./output", type=str)
    # Stop words list
    parser.add_argument(
        "--stop_words",
        default=["</s>", "<|im_end|>", "<|endoftext|>", "\n题目："],
        type=list,
    )
    parser.add_argument("--dataset", default=None, type=str)
    args = parser.parse_args()
    return args


def prepare_data(data_name, args):
    # Load current JSON file using load_data_vanilla
    examples = load_data_vanilla(args.input_path)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    os.makedirs(f"{output_dir}/{args.exp_name}/{data_name}", exist_ok=True)

    # If there are sample deduplication or filtering operations, they can be implemented here
    processed_samples = []
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def calculate_token_statistics(examples):
    """
    Calculate token statistics from examples
    """
    output_tokens = []
    prefill_tokens = []
    total_tokens = []
    
    for example in examples:
        if "output_tokens" in example:
            output_tokens.append(example["output_tokens"])
        if "prefill_tokens" in example:
            prefill_tokens.append(example["prefill_tokens"])
        if "total_tokens" in example:
            total_tokens.append(example["total_tokens"])
    
    stats = {}
    if output_tokens:
        stats["avg_output_tokens"] = sum(output_tokens) / len(output_tokens)
        stats["total_samples"] = len(output_tokens)
        stats["min_output_tokens"] = min(output_tokens)
        stats["max_output_tokens"] = max(output_tokens)
    
    if prefill_tokens:
        stats["avg_prefill_tokens"] = sum(prefill_tokens) / len(prefill_tokens)
        stats["min_prefill_tokens"] = min(prefill_tokens)
        stats["max_prefill_tokens"] = max(prefill_tokens)
    
    if total_tokens:
        stats["avg_total_tokens"] = sum(total_tokens) / len(total_tokens)
        stats["min_total_tokens"] = min(total_tokens)
        stats["max_total_tokens"] = max(total_tokens)
    
    return stats


def main(data_name, args):
    """
    Process and evaluate a single JSON file
    data_name is the math dataset name (here using JSON filename without extension)
    """
    examples, processed_samples = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # Calculate token statistics
    token_stats = calculate_token_statistics(examples)
    print(f"Token Statistics for {data_name}:")
    for key, value in token_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Initialize python executor, determine answer retrieval method based on prompt_type
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for cnt, example in tqdm(enumerate(examples), total=len(examples)):
        # For different datasets, the answer field name may be different
        if args.exp_name.lower().find("omni-math") != -1:
            example["solution"] = example["answer"]
        else:
            try:
                example["solution"] = example["solution"]
            except:
                example["solution"] = example["answer"]

        idx = example.get("idx", cnt)

        try:
            example["question"] = example["question"]
        except:
            example["question"] = example["problem"]

        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
        }

        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
            "domain",
            "difficulty",
            "source",
            "output_tokens",  # 添加token相关字段
            "prefill_tokens",
            "total_tokens",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    codes = []
    for i in range(len(examples)):
        code = examples[i]["output"]
        for stop_word in args.stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]

    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i]
        result = results[i]
        preds = [result[0]]
        reports = [result[1]]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # Add token statistics to result_json
    result_json.update(token_stats)

    # Modify output filename to include scale (size) and method to avoid overwriting
    out_dir = os.path.join(args.output_dir, args.exp_name, data_name)
    os.makedirs(out_dir, exist_ok=True)
    # Here using size-method as part of the filename, can be adjusted as needed
    out_file = os.path.join(
        out_dir,
        f"{getattr(args, 'size', 'default')}-{getattr(args, 'method', 'default')}_math_eval.jsonl",
    )
    save_jsonl(all_samples, out_file)

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


def main_all(args):
    """
    Traverse all JSON/JSONL files in the specified folder, call main() for each file for evaluation,
    and summarize the results in a CSV file.
    """
    # Only read all files in the base_dir directory (don't traverse subdirectories)
    json_files = []
    for file in os.listdir(args.base_dir):
        filepath = os.path.join(args.base_dir, file)
        if os.path.isfile(filepath) and (
            file.endswith(".json") or file.endswith(".jsonl")
        ):
            json_files.append(filepath)

    if not json_files:
        print("No JSON/JSONL files found in the folder.")
        return

    results_table = {}  # Store all metrics including token statistics
    for json_file in json_files:
        # Use filename (without extension) as dataset name
        dataset = args.dataset
        args.input_path = json_file
        # Optional: Set some default values for output filenames to distinguish results from different files
        args.size = "default"
        args.method = dataset
        print(f"Processing: dataset={dataset} from file {json_file}")
        
        result_json = main(dataset, args)
        results_table[json_file] = result_json


    # Construct CSV table with additional token statistics
    output_csv = os.path.join(args.output_dir, "all_results.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Create header with all possible metrics
        header = ["Dataset", "Accuracy", "Avg_Output_Tokens", "Avg_Prefill_Tokens", 
                 "Avg_Total_Tokens", "Total_Samples", "Min_Output_Tokens", 
                 "Max_Output_Tokens"]
        writer.writerow(header)
        
        for dataset, metrics in sorted(results_table.items()):
            row = [
                dataset,
                metrics.get("acc", "N/A"),
                f"{metrics.get('avg_output_tokens', 0):.2f}" if metrics.get('avg_output_tokens') else "N/A",
                f"{metrics.get('avg_prefill_tokens', 0):.2f}" if metrics.get('avg_prefill_tokens') else "N/A",
                f"{metrics.get('avg_total_tokens', 0):.2f}" if metrics.get('avg_total_tokens') else "N/A",
                metrics.get('total_samples', "N/A"),
                metrics.get('min_output_tokens', "N/A"),
                metrics.get('max_output_tokens', "N/A"),
            ]
            writer.writerow(row)
    
    print(f"All evaluation results have been saved to {output_csv}")
    
    # Also create a summary statistics file
    summary_csv = os.path.join(args.output_dir, "token_summary.csv")
    with open(summary_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        
        # Calculate overall statistics across all datasets
        all_avg_output_tokens = [metrics.get('avg_output_tokens', 0) for metrics in results_table.values() if metrics.get('avg_output_tokens')]
        all_total_samples = [metrics.get('total_samples', 0) for metrics in results_table.values() if metrics.get('total_samples')]
        
        if all_avg_output_tokens:
            writer.writerow(["Overall_Avg_Output_Tokens", f"{sum(all_avg_output_tokens) / len(all_avg_output_tokens):.2f}"])
        if all_total_samples:
            writer.writerow(["Total_Samples_All_Datasets", sum(all_total_samples)])
        writer.writerow(["Total_Datasets", len(results_table)])
    
    print(f"Token summary statistics have been saved to {summary_csv}")


if __name__ == "__main__":
    args = parse_args()
    # Call main_all to traverse all JSON/JSONL files in the specified folder and summarize results
    main_all(args)