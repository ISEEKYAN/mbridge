# Example to use tp/pp/cp/vpp to test dense model
# torchrun --nproc_per_node=8 example/qwen3vl/load_model_and_forward.py --model_path /path/to/model

import os
import argparse
import requests

try:
    from transformers import Qwen3VLProcessor
except:
    print(f"your install the tranformers>=4.57.0 or install from source")

import torch
import torch.nn.functional as F

from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
)
from megatron.core.models.gpt.gpt_model import ModelType

from mbridge import AutoBridge


def download_img(filename):
    image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"downlaod fail: {e}")
        raise e


def get_sample_for_forward(hf_model_path):
    processor = Qwen3VLProcessor.from_pretrained(hf_model_path)
    # text = "Please describe this picture completely and in detail, including the details, characters, scenes, etc."
    text = "Describe this image in shortly."
    filename = "../australia.jpg"
    if True:
        if not os.path.exists("../australia.jpg"):
            download_img(filename)
        messages = [{
            "role":
            "user",
            "content": [{
                "type": "image",
                "image": filename
            }, {
                "type": "text",
                "text": text
            }]
        }]
    else:
        text = "Given the accelerating trajectory of artificial intelligence, where do you foresee the most critical point of divergence between a future in which AI acts as a fundamentally benevolent, symbiotic partner in elevating human consciousness, collective intelligence, and our capacity to solve existential challenges, and a future where it inadvertently becomes an insidious, alienating force that amplifies societal biases, erodes human agency, and creates a new, opaque class structure based on access to and control of cognitive capital—and what specific, measurable factors in our current approach to AI development, governance, and education will be the primary determinants in steering us toward one outcome over the other?"
        messages = [{
            "role":
            "user",
            "content": [{
                "type": "text",
                "text": text
            }]
        }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        max_pixels=256 * 28 * 28,
    )
    inputs.pop("token_type_ids", None)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    return inputs


# hf logits vs megatron logits
def cos_similarity(a, b):
    print(f"a {a.shape} b {b.shape}")
    a = a.float()
    #a = a / a.norm(dim=-1, keepdim=True)
    a = torch.exp(a - a.max(dim=-1, keepdim=True)[0])
    a = a / a.norm(dim=-1, keepdim=True)
    """
    a = (a - a.mean(dim=-1, keepdim=True)) 
    a = a / a.norm(dim=-1, keepdim=True)
    """
    b = b.float()
    #b =  b / b.norm(dim=-1, keepdim=True)
    b = torch.exp(b - b.max(dim=-1, keepdim=True)[0])
    b = b / b.norm(dim=-1, keepdim=True)
    """
    b = (b - b.mean(dim=-1, keepdim=True)) 
    b =  b / b.norm(dim=-1, keepdim=True)
    """
    sim = (a * b).sum(dim=-1)
    print(f"hf vs megatron cos_similarity min: {sim.min()}; max: {sim.max()}; mean: {sim.mean()}")


def init_distributed(tp=2, pp=1, cp=1, vpp=1, ep=1, etp=None):
    """Initialize distributed environment"""
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(torch.distributed.get_rank() % 8)
    if pp <= 1:
        vpp = None
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
    )
    model_parallel_cuda_manual_seed(0)


def get_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load model and generate text")
    parser.add_argument(
        "--model_path", type=str, required=True, help="HuggingFace model path"
    )
    parser.add_argument("--tp", type=int, default=2, help="Tensor model parallel size")
    parser.add_argument(
        "--pp", type=int, default=1, help="Pipeline model parallel size"
    )
    parser.add_argument("--cp", type=int, default=1, help="Context parallel size")
    parser.add_argument(
        "--vpp", type=int, default=None, help="Virtual pipeline model parallel size"
    )
    parser.add_argument("--ep", type=int, default=1, help="Expert model parallel size")
    parser.add_argument(
        "--etp", type=int, default=None, help="Expert tensor parallel size"
    )
    parser.add_argument("--check_export", action="store_true", help="Trust remote code")
    args = parser.parse_args()
    return args


def mcore_fwd_fn(data_iterator, model):
    sample = next(data_iterator)

    output_tensor = model(
        input_ids=sample["input_ids"].cuda(),
        position_ids=None,
        attention_mask=None,
        pixel_values=sample["pixel_values"].cuda() if "pixel_values" in sample else None,
        image_grid_thw=sample["image_grid_thw"].cuda() if "image_grid_thw" in sample else None,
    )
    if isinstance(output_tensor, tuple):
        output_tensor = output_tensor[0]
    assert isinstance(output_tensor, torch.Tensor)

    def loss_fn(output_tensor, non_loss_data=True):
        loss = output_tensor.mean()
        return loss, {
            'loss': loss.detach(),
            'logits': output_tensor.detach(),
        }

    return output_tensor, loss_fn


def main():
    args = get_args()
    print(f"{args=}")

    # Initialize distributed environment
    init_distributed(
        tp=args.tp,
        pp=args.pp,
        cp=args.cp,
        vpp=args.vpp,
        ep=args.ep,
        etp=args.etp,
    )

    # Load megatron model
    hf_model_path = args.model_path
    print(f"rank{torch.distributed.get_rank()}: start loading model ...")
    bridge = AutoBridge.from_pretrained(hf_model_path)
    bridge.config.sequence_parallel = True
    if args.pp > 1:
        num_layer = bridge.hf_config.text_config.num_hidden_layers
        first_last_layer = num_layer - (num_layer + args.pp - 1) // args.pp * (args.pp - 2)
        assert first_last_layer > 1
        bridge.set_extra_args(
            num_layers_in_first_pipeline_stage=first_last_layer // 2,
            num_layers_in_last_pipeline_stage=(first_last_layer + 1) // 2,
        )
    model = bridge.get_model(model_type=ModelType.encoder_and_decoder)
    assert len(model) == 1
    bridge.load_weights(model, hf_model_path, memory_efficient=True)

    # check the export
    if args.check_export:
        print(f"rank{torch.distributed.get_rank()}: end load weight, start check export ...")
        keys = bridge.safetensor_io.load_hf_weight_names()
        loaded_keys = set()
        # export weights
        for k, v in bridge.export_weights(model):
            gt = bridge.safetensor_io.load_one_hf_weight(k).cuda()
            assert v.shape == gt.shape, f"mismatch of {k}"
            assert torch.equal(v, gt), f"mismatch of {k}"
            loaded_keys.add(k)
        assert len(bridge.export_weights_buff) == 0

        missing_keys = set(keys) - loaded_keys
        missing_keys = sorted(list(missing_keys))
        assert len(missing_keys) == 0
        print(f"missing keys: {missing_keys}")

    print(f"rank{torch.distributed.get_rank()}: end load weight, start forward ...")

    sample = get_sample_for_forward(hf_model_path)
    real_seq_length = sample["input_ids"].shape[-1]
    torch.distributed.barrier()
    with torch.no_grad():
        fwd_bwd_function = get_forward_backward_func()

        seq_length = real_seq_length
        if real_seq_length % args.tp != 0:
            seq_length = (real_seq_length + args.tp - 1) // args.tp * args.tp
            sample["input_ids"] = F.pad(
                sample["input_ids"],
                (0, seq_length - real_seq_length, 0, 0),
                value=0,
            )

        mcore_output = fwd_bwd_function(
            forward_step_func=mcore_fwd_fn,
            data_iterator=iter([sample]),
            model=model,
            num_microbatches=1,
            forward_only=True,
            seq_length=seq_length,
            decoder_seq_length=seq_length,
            micro_batch_size=1,
        )

        if mpu.is_pipeline_last_stage():
            megatron_output = mcore_output[0]["logits"]
            if mpu.get_tensor_model_parallel_world_size() > 1:
                megatron_output = gather_from_tensor_model_parallel_region(megatron_output)

            megatron_output = megatron_output[:, :real_seq_length, :]
            hf_output = torch.load("/tmp/hf_qwen3vl.pt",
                                   map_location="cpu").to(megatron_output.device)
            cos_similarity(hf_output, megatron_output)
            print(f"Finish Done")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
