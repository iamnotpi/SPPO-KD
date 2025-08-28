from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams

import argparse
import torch
import json
import os
from pathlib import Path
import random
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2"
    )
    parser.add_argument("--output_dir", type=str, default="generated/iter1")
    parser.add_argument("--prompts", type=str, default=os.environ.get("HF_PROMPTS_DATASET", "UCLA-AGI/data-mistral-7b-instruct-sppo-iter1"))
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--frac_len", type=int, default=0)
    parser.add_argument("--data_frac", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    return parser.parse_args()


def apply_template(text, tokenizer):
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}, {"role": "assistant", "content": "None"}],
            tokenize=False, add_generate_prompt=True
        ).split("None")[0]
    else:
        return text


def split_prompts(prompts, frac_len, data_frac):
    if frac_len > 0:
        split_len = frac_len
        if split_len * (data_frac + 1) > len(prompts):
            return prompts[split_len * data_frac:]
        else:
            return prompts[split_len * data_frac: split_len * (data_frac + 1)]
    else:
        return prompts[:]


def load_prompts_source(source):
    if os.path.isdir(source):
        candidates = [
            "train.jsonl", "train.json", "data.jsonl", "data.json",
            "train.parquet", "data.parquet"
        ]
        for name in candidates:
            p = os.path.join(source, name)
            if os.path.exists(p):
                if p.endswith(".jsonl") or p.endswith(".json"):
                    return load_dataset("json", data_files=p, split="train")
                if p.endswith(".parquet"):
                    return load_dataset("parquet", data_files=p, split="train")
    else:
        if source.endswith(".jsonl") or source.endswith(".json"):
            return load_dataset("json", data_files=source, split="train")
        if source.endswith(".parquet"):
            return load_dataset("parquet", data_files=source, split="train")
    return load_dataset(source, split="train")

def main():
    args = parse_arguments()
    model_path = args.model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_prompts_source(args.prompts)

    if "mistral" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    elif "llama-3" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    elif "gemma-2" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    elif "gpt2" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    else:
        # Fallback: try to load tokenizer from the provided model path (e.g., local GPT-2 folder)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    # import pdb
    # pdb.set_trace()
    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=args.world_size,
        )
        use_vllm = True
    except Exception as e:
        print(f"vLLM failed to initialize for {model_path}: {e}. Falling back to transformers generate().")
        use_vllm = False
    prompts = [apply_template(data[idx]["prompt"], tokenizer) for idx in range(len(data))]
    print(prompts[0])
    data_frac, frac_len = args.data_frac, args.frac_len
    prompts = split_prompts(prompts, frac_len, data_frac)

    pairs = args.pairs

    os.makedirs(args.output_dir, exist_ok=True)

    for p in range(pairs):
        set_seed(p * 50)
        if use_vllm:
            sampling_params = SamplingParams(
                temperature=1.0,
                top_p=1.0,
                max_tokens=args.maxlen,
                seed=p * 50,
            )
            response = llm.generate(prompts, sampling_params)
            output = list(map(lambda x: x.outputs[0].text, response))
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            output = []
            for pr in prompts:
                inputs = tokenizer(pr, return_tensors="pt").to(device)
                gen = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    max_new_tokens=args.maxlen,
                    pad_token_id=tokenizer.eos_token_id,
                )
                text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                output.append(text)
        with open(f"{args.output_dir}/responses_{data_frac}_{p}.json", "w") as f:
            json.dump(output, f)


if __name__ == "__main__":
    main()
