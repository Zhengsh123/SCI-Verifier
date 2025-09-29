import argparse
import json
import os
import re
import sys
import time
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from prompts import PROMPTS
from collections import defaultdict

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def process_judgment(judgment_str: str) -> str:
    for char in judgment_str[::-1]:
        if char in ("A", "B", "C"):
            return char
    return ""


def extract_judgment(raw_judgment):

    matches = re.findall(r"\\boxed{([A-Z])}", raw_judgment)
    if matches:
        return matches[-1]

    if re.search(r"\byes\b", raw_judgment, re.IGNORECASE):
        return "A"
    elif re.search(r"\bno\b", raw_judgment, re.IGNORECASE):
        return "B"
    
    if re.search(r"\bincorrect\b", raw_judgment, re.IGNORECASE):
        return "B"
    elif re.search(r"\bcorrect\b", raw_judgment, re.IGNORECASE):
        return "A"

    return None


def extract_xverify_answer(text: str) -> str:
    text_upper = text.upper().strip()
    if "INCORRECT" in text_upper:
        return "B"
    elif "CORRECT" in text_upper:
        return "A"
    else:
        return extract_judgment(text)


def extract_answer(test_result: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)\}", test_result)
    if matches:
        return matches[-1]
    return test_result


def batch_process_samples(args, tokenizer, vllm_model, sampling_params, samples):
    batch_prompt = PROMPTS[args.prompt_type]
    prompts = []
    for s in samples:
        llm_resp = s["llm_response"]
        if len(llm_resp) > args.max_response_chars:
            llm_resp = llm_resp[-args.max_response_chars:]

        prompt = batch_prompt.format(
            question=s["question"],
            gold_answer=s["gold_answer"],
            llm_response=llm_resp,
        )
        prompts.append(prompt)

    encodings = tokenizer(prompts, return_length=True, add_special_tokens=False)
    lengths = encodings["length"]

    batch_inputs, valid_samples, results = [], [], []

    for s, l, p in zip(samples, lengths, prompts):
        if l > 32768:
            results.append(
                {
                    "question": s["question"],
                    "gold_answer": s["gold_answer"],
                    "llm_response": s["llm_response"], 
                    "gold_judgment": s["gold_judgment"],
                    "raw_judgment": "[SKIPPED: too long]",
                    "final_judgment": "C",
                    "correct": 0,
                    "domain": s.get("domain",""),
                    "source": s.get("source",""),
                    "aug":s.get("aug",""),
                }
            )
        else:
            msg = [{"role": "user", "content": p}]
            model_input = tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False,
            )
            valid_samples.append(s)
            batch_inputs.append(model_input)

    if not batch_inputs:
        return results

    outputs = vllm_model.generate(batch_inputs, sampling_params)

    for s, output in zip(valid_samples, outputs):
        raw_judgment = output.outputs[0].text.strip()

        if args.prompt_type == "cot":
            final_judgement = extract_judgment(raw_judgment)
        elif args.prompt_type == "instruct":
            final_judgement = extract_judgment(raw_judgment)
        elif args.prompt_type == "short":
            final_judgement = process_judgment(raw_judgment)
        elif args.prompt_type == "xverify":
            final_judgement = extract_xverify_answer(raw_judgment)
        else:
            final_judgement = "C"
        
        
        gold_judgement = s["gold_judgment"]

        if (final_judgement in ("B", "C")) and not gold_judgement:
            correctness = 1
        elif final_judgement == "A" and gold_judgement:
            correctness = 1
        else:
            correctness = 0

        results.append(
            {
                'uid':s.get('uid',''),
                "question": s["question"],
                "gold_answer": s["gold_answer"],
                "llm_response": s["llm_response"],
                "gold_judgment": gold_judgement,
                "raw_judgment": raw_judgment,
                "final_judgment": final_judgement,
                "correct": correctness,
                "domain": s.get("domain",""),
                "source": s.get("source",""),
                "aug":s.get("aug",""),
            }
        )

    return results

def compute_accuracy_by_domain(all_results):
    domain_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    # 遍历结果
    for item in all_results:
        domain = item.get("domain", "unknown")
        is_correct = item.get("correct", False)
        
        domain_stats[domain]["total"] += 1
        if is_correct:
            domain_stats[domain]["correct"] += 1
    
    # 计算正确率
    domain_accuracy = {}
    for domain, stats in domain_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        accuracy = correct / total if total > 0 else 0
        domain_accuracy[domain] = {
            "correct": correct,
            "total": total,
            "accuracy": round(accuracy, 4)
        }
    
    return domain_accuracy

def main(args,data_path):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    vllm_model = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_num_seqs=args.batch_size * 4,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=1,
        top_p=1.0,
        stop=args.stop_tokens,
    )

    dataset=load_jsonl(data_path)

    all_results = []
    total_correct = 0

    batches = [
        dataset[i : i + args.batch_size]
        for i in range(0, len(dataset), args.batch_size)
    ]

    time_start = time.time()
    with tqdm(total=len(batches), desc="Processing batches") as pbar:
        for batch in batches:
            batch_results = batch_process_samples(
                args, tokenizer, vllm_model, sampling_params, batch
            )
            all_results.extend(batch_results)
            batch_correct = sum(r["correct"] for r in batch_results)
            total_correct += batch_correct
            pbar.update(1)
            pbar.set_postfix({"Acc": f"{total_correct/len(all_results):.2%}"})
    time_end = time.time()

    accuracy = total_correct / len(all_results) if all_results else 0
    domain_acc=compute_accuracy_by_domain(all_results)
    total_tokens = 0
    for r in all_results:
        tokens = tokenizer.encode(r["raw_judgment"], add_special_tokens=False)
        total_tokens += len(tokens)
    avg_tokens = total_tokens / len(all_results) if all_results else 0

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = f"{args.output_dir}/results_{os.path.basename(data_path)}"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(f"{args.output_dir}/summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Data: {data_path}\n")
        f.write(f"GPU Count: {args.tensor_parallel_size}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Samples: {len(all_results)}\n")
        f.write(f"Correct: {total_correct}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Avg raw_judgment tokens: {avg_tokens:.2f}\n")
        f.write(f"Time: {time_end-time_start:.4f}\n\n")

        f.write("Per-domain accuracy:\n")
        for domain, stats in domain_acc.items():
            f.write(f"  {domain}: {stats['accuracy']:.4f}\n")

    print("\n" + "=" * 50)
    print("Evaluation Summary:")
    print(f"Model: {args.model_path}")
    print(f"Data: {data_path} | Samples: {len(all_results)}")
    print(
        f"Configuration: GPU={args.tensor_parallel_size}, Batch={args.batch_size}, Workers=N/A"
    )
    print(f"Results: Correct {total_correct}/{len(all_results)} | Accuracy: {accuracy:.4f}")
    print(f"Avg raw_judgment tokens: {avg_tokens:.2f}")
    print("Per-domain accuracy:")
    for domain, stats in domain_acc.items():
        print(f"  {domain}: {stats['accuracy']:.4f}")
    print(f"Results saved to: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU vLLM Batch Evaluation")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True,
                    help="Path to the folder containing datasets")
    parser.add_argument("--dataset_name", type=str, required=True,
                    help="Name of the dataset (without .jsonl extension)")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--prompt_type",
        type=str,
        choices=["instruct", "cot", "xverify","short"],
        default="cot",
        help="Choose the prompt format: instruct, cot, or xverify",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--stop_tokens", nargs="+", default=None)

    parser.add_argument(
        "--max_response_chars",
        type=int,
        default=16384,
        help="Truncate llm_response to last N characters if too long",
    )

    args = parser.parse_args()

    data_path = os.path.join(args.data_root, args.dataset_name + ".jsonl")
    
    if not os.path.exists(args.model_path):
        sys.exit(f"Model path '{args.model_path}' does not exist")
    if not os.path.exists(data_path):
        sys.exit(f"Data path '{data_path}' does not exist")

    file_name = args.prompt_type
    args.output_dir = os.path.join(
        args.output_dir,
        args.dataset_name,
        args.model_path.split("/")[-1],
        file_name,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    main(args,data_path)
