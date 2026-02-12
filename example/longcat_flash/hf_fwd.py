# Generate HF reference output for the LongCat Flash model
# This script runs the model using HuggingFace Transformers and saves the output
# for comparison with the Megatron forward pass.
#
# Usage:
#   python example/longcat_flash/hf_fwd.py --model_path /path/to/longcat_flash
#
# Note: For the full 560B model, this requires multiple GPUs via device_map="auto".
# For smaller variants, a single GPU may suffice.

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HF_OUTPUT_PATH = "/tmp/hf_longcat_flash.pt"


def main():
    parser = argparse.ArgumentParser(
        description="Run HF forward pass for LongCat Flash and save output"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="HuggingFace model path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A bubble sort in python is ",
        help="Prompt for forward pass",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(model.device)
    print(f"Input shape: {input_ids.shape}")

    with torch.no_grad():
        output = model(input_ids=input_ids)
        logits = output.logits.cpu()

    print(f"Output shape: {logits.shape}")
    print(
        f"Output stats: mean={logits.float().mean():.4f}, "
        f"std={logits.float().std():.4f}"
    )

    torch.save(logits, HF_OUTPUT_PATH)
    print(f"Saved HF output to {HF_OUTPUT_PATH}")


if __name__ == "__main__":
    main()