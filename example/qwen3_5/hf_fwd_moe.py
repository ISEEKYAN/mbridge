import argparse

import torch

try:
    from transformers import Qwen3_5MoeForConditionalGeneration
except:
    print(f"your install the tranformers>=5.2.0 or install from source")

from example.qwen3_5.load_model_and_forward import get_sample_for_forward


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load model and generate text")
    parser.add_argument(
        "--model_path", type=str, required=True, help="HuggingFace model path"
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        default="image",
        choices=["image", "video", "mix"],
        help="sample type",
    )
    args = parser.parse_args()

    # default: Load the model on the available device(s)
    torch.set_grad_enabled(False)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="cuda:0",
    )

    for pname, params in model.named_parameters():
        print(f"Model weight {pname=} {params.shape} {params.dtype} {params.sum()}")

    # Preparation for inference
    inputs = get_sample_for_forward(args.model_path, args.sample_type)

    for k in inputs:
        inputs[k] = inputs[k].cuda()

    # Inference: Generation of the output
    hf_output = model.forward(**inputs)

    print(hf_output.logits.shape, hf_output.logits.device, hf_output.logits.dtype)
    torch.save(hf_output.logits.cpu(), "qwen3p5_save/hf_qwen3_5.pt")

    print(f"hf Done")
