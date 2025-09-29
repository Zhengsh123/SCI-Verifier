import os
import re
import sys
import time
import json
import argparse
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
from threading import Lock
from openai import OpenAI

from prompts import PROMPTS  

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def write_jsonl(path, records, mode="a"):
    if not isinstance(records, list):
        records = [records]
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

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
    return raw_judgment[-1:]

def extract_xverify_answer(text: str) -> str:
    text_upper = text.upper().strip()
    if "INCORRECT" in text_upper:
        return "B"
    elif "CORRECT" in text_upper:
        return "A"
    else:
        return "C"

class ChatModel:
    def __init__(self, api_key: str, base_url: str, model_name: str, generation_kwargs=None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.generation_kwargs = generation_kwargs or {}

    def call_api(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.generation_kwargs
        )
        return response.choices[0].message.content.strip()

def process_sample(chat_model: ChatModel, s, args):
    # prompt
    batch_prompt = PROMPTS[args.prompt_type]
    llm_resp = s["llm_response"]
    if len(llm_resp) > args.max_response_chars:
        llm_resp = llm_resp[-args.max_response_chars:]

    prompt = batch_prompt.format(
        question=s["question"],
        gold_answer=s["gold_answer"],
        llm_response=llm_resp,
    )
    messages = [{"role": "user", "content": prompt}]
    raw_judgment = chat_model.call_api(messages)
    if args.prompt_type == "cot":
        final_judgement = extract_judgment(raw_judgment)
    elif args.prompt_type == "instruct":
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

    result = {
        "question": s["question"],
        "gold_answer": s["gold_answer"],
        "llm_response": s["llm_response"],
        "gold_judgment": s["gold_judgment"],
        "raw_judgment": raw_judgment,
        "final_judgment": final_judgement,
        "correct": correctness,
        "domain": s.get("domain", ""),
        "source": s.get("source", ""),
        "aug": s.get("aug", ""),
    }
    return result

def compute_accuracy_by_domain(all_results):
    domain_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in all_results:
        domain = item.get("domain", "unknown")
        is_correct = item.get("correct", False)
        domain_stats[domain]["total"] += 1
        if is_correct:
            domain_stats[domain]["correct"] += 1

    domain_accuracy = {}
    for domain, stats in domain_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        acc = correct / total if total > 0 else 0
        domain_accuracy[domain] = {
            "correct": correct,
            "total": total,
            "accuracy": round(acc, 4),
        }
    return domain_accuracy

def main(args, data_path):
    generation_kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stop": args.stop_tokens,
    }

    chat_model = ChatModel(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        generation_kwargs=generation_kwargs,
    )

    dataset = load_jsonl(data_path)
    all_results, total_correct = [], 0

    batches = [dataset[i:i + args.batch_size] for i in range(0, len(dataset), args.batch_size)]
    time_start = time.time()

    with tqdm(total=len(batches), desc="Processing batches") as pbar:
        for batch in batches:
            results_in_batch = [None] * len(batch)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {executor.submit(process_sample, chat_model, s, args): idx
                           for idx, s in enumerate(batch)}
                for fut in concurrent.futures.as_completed(futures):
                    idx = futures[fut]
                    res = fut.result()
                    results_in_batch[idx] = res

            for r in results_in_batch:
                if not r:
                    continue
                all_results.append(r)
                total_correct += r.get("correct", 0)

            pbar.update(1)
            if all_results:
                pbar.set_postfix({"Acc": f"{total_correct/len(all_results):.2%}"})

    time_end = time.time()
    accuracy = total_correct / len(all_results) if all_results else 0
    domain_acc = compute_accuracy_by_domain(all_results)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = f"{args.output_dir}/results_{os.path.basename(data_path)}"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(f"{args.output_dir}/summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Data: {data_path}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Samples: {len(all_results)}\n")
        f.write(f"Correct: {total_correct}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Time: {time_end-time_start:.4f}\n\n")

        f.write("Per-domain accuracy:\n")
        for domain, stats in domain_acc.items():
            f.write(f"  {domain}: {stats['accuracy']:.4f}\n")

    print("\n" + "=" * 50)
    print("Evaluation Summary:")
    print(f"Model: {args.model_name}")
    print(f"Data: {data_path} | Samples: {len(all_results)}")
    print(f"Configuration: Batch={args.batch_size}, Workers={args.max_workers}")
    print(f"Results: Correct {total_correct}/{len(all_results)} | Accuracy: {accuracy:.4f}")
    print("Per-domain accuracy:")
    for domain, stats in domain_acc.items():
        print(f"  {domain}: {stats['accuracy']:.4f}")
    print(f"Results saved to: {output_path}")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel API Evaluation with Summary")
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--base_url", type=str,)
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--prompt_type", type=str, choices=["instruct", "cot", "xverify"], default="cot")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_workers", type=int, default=64)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--stop_tokens", nargs="+", default=None)
    parser.add_argument("--max_response_chars", type=int, default=16384)

    args = parser.parse_args()

    data_path = os.path.join(args.data_root, args.dataset_name + ".jsonl")
    if not os.path.exists(data_path):
        sys.exit(f"Data path '{data_path}' does not exist")

    args.output_dir = os.path.join(args.output_dir, args.dataset_name, args.model_name, args.prompt_type)
    os.makedirs(args.output_dir, exist_ok=True)

    main(args, data_path)
