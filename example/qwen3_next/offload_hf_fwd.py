# Example to use tp/pp/cp/vpp to test dense model
# python3 offload_hf_fwd.py --model_path /path/to/model


import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser(description="Load model and generate text")
    parser.add_argument(
        "--model_path", type=str, required=True, help="HuggingFace model path"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save weights"
    )
    args = parser.parse_args()
    return args


def load_hf_model_and_forward(args):
    hf_model_path = args.model_path
    hf_model = AutoModelForCausalLM.from_pretrained(
            hf_model_path,
            dtype="auto",
            device_map="auto",)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    prompt = "李白，字太白，号"
    messages = [
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    for pname, params in hf_model.named_parameters():
        print(f"Trace export_weights name={pname}: {params.shape=} {params.dtype=} {params.sum()}")

    model_inputs = tokenizer([text], return_tensors="pt").to(hf_model.device)

    with torch.no_grad():
        hf_output = hf_model(
            **model_inputs)
        print(f"rank hf_output {hf_output.logits}")
        torch.save(hf_output.logits, "./hf_qwen3next.pt")


if __name__ == "__main__":
    args = get_args()
    load_hf_model_and_forward(args)
