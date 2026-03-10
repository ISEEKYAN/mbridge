# Example to use tp/pp/cp/vpp to test dense model
# torchrun --nproc_per_node=8 example/qwen3_omni/load_model_and_forward.py --model_path /path/to/model

import argparse
import os

try:
    from transformers import Qwen3OmniMoeProcessor
except:
    print(f"your install the tranformers>=4.57.0 or install from source")

import torch
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_model import ModelType
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from mbridge import AutoBridge


def get_sample_for_forward(hf_model_path):
    processor = Qwen3OmniMoeProcessor.from_pretrained(hf_model_path)
    image_file = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"
    audio_file = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_file},
                {"type": "audio", "audio": audio_file},
                {"type": "text", "text": "Describe those audios and images in shortly and respectively."},
            ],
        },
    ]

    kwargs = {
        "videos_kwargs": {
            "use_audio_in_video": False,
        },
    }
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_vision_id=True,  # have better to add this
        **kwargs,
    )

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    if "pixel_values_videos" in inputs:
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(torch.bfloat16)
        print(f'{inputs["pixel_values_videos"].shape=}')
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)

    return inputs


def gather_output_from_cp(input_: torch.Tensor, seq_dim, cp_size, cp_group):
    assert seq_dim in [0, 1] and input_.dim() > seq_dim
    # Split local_logits into two parts
    input_ = input_.view(
        *input_.shape[0:seq_dim],
        2,
        input_.shape[seq_dim] // 2,
        *input_.shape[(seq_dim + 1) :],
    )

    gathered_logits = [torch.zeros_like(input_) for _ in range(cp_size)]
    torch.distributed.all_gather(gathered_logits, input_, group=cp_group)

    reorded_logits = [None for _ in range(2 * cp_size)]
    if seq_dim == 1:
        for rank in range(cp_size):
            reorded_logits[rank] = gathered_logits[rank][:, 0]
            reorded_logits[2 * cp_size - rank - 1] = gathered_logits[rank][:, 1]
    elif seq_dim == 0:
        for rank in range(cp_size):
            reorded_logits[rank] = gathered_logits[rank][0]
            reorded_logits[2 * cp_size - rank - 1] = gathered_logits[rank][1]
    else:
        assert False
    gathered_logits = torch.cat(reorded_logits, dim=seq_dim)

    return gathered_logits


# hf logits vs megatron logits
def cos_similarity(a, b):
    print(f"a {a.shape} b {b.shape}")
    a = a.float()
    # a = a / a.norm(dim=-1, keepdim=True)
    a = torch.exp(a - a.max(dim=-1, keepdim=True)[0])
    a = a / a.norm(dim=-1, keepdim=True)
    """
    a = (a - a.mean(dim=-1, keepdim=True)) 
    a = a / a.norm(dim=-1, keepdim=True)
    """
    b = b.float()
    # b =  b / b.norm(dim=-1, keepdim=True)
    b = torch.exp(b - b.max(dim=-1, keepdim=True)[0])
    b = b / b.norm(dim=-1, keepdim=True)
    """
    b = (b - b.mean(dim=-1, keepdim=True)) 
    b =  b / b.norm(dim=-1, keepdim=True)
    """
    sim = (a * b).sum(dim=-1)
    print(
        f"hf vs megatron cos_similarity min: {sim.min()}; max: {sim.max()}; mean: {sim.mean()}"
    )


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
        input_features= (
            sample["input_features"].cuda() if "input_features" in sample else None
        ),
        feature_attention_mask = (
            sample["feature_attention_mask"].cuda() if "feature_attention_mask" in sample else None
        ),
        position_ids=None,
        attention_mask=None,
        pixel_values=(
            sample["pixel_values"].cuda() if "pixel_values" in sample else None
        ),
        image_grid_thw=(
            sample["image_grid_thw"].cuda() if "image_grid_thw" in sample else None
        ),
        pixel_values_videos=(
            sample["pixel_values_videos"].cuda()
            if "pixel_values_videos" in sample
            else None
        ),
        video_grid_thw=(
            sample["video_grid_thw"].cuda() if "video_grid_thw" in sample else None
        ),
    )
    if isinstance(output_tensor, tuple):
        output_tensor = output_tensor[0]
    assert isinstance(output_tensor, torch.Tensor)

    def loss_fn(output_tensor, non_loss_data=True):
        loss = output_tensor.mean()
        return loss, {
            "loss": loss.detach(),
            "logits": output_tensor.detach(),
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
        first_last_layer = num_layer - (num_layer + args.pp - 1) // args.pp * (
            args.pp - 2
        )
        assert first_last_layer > 1
        bridge.set_extra_args(
            num_layers_in_first_pipeline_stage=first_last_layer // 2,
            num_layers_in_last_pipeline_stage=(first_last_layer + 1) // 2,
        )
    model = bridge.get_model(model_type=ModelType.encoder_and_decoder)
    assert len(model) == 1

    bridge.load_weights(model, hf_model_path, memory_efficient=False)

    # check the export
    if args.check_export:
        print(
            f"rank{torch.distributed.get_rank()}: end load weight, start check export ..."
        )
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
        thinker_missing_keys = []
        for key in missing_keys:
            if key.startswith("thinker."):
                thinker_missing_keys.append(key)
        assert len(thinker_missing_keys) == 0, f"missing keys: {thinker_missing_keys}"

    print(f"rank{torch.distributed.get_rank()}: end load weight, start forward ...")

    sample = get_sample_for_forward(hf_model_path)
    input_mask = sample["input_ids"] != bridge.hf_config.image_token_id
    input_mask = input_mask & sample["input_ids"] != bridge.hf_config.video_token_id
    input_mask = input_mask & (sample["input_ids"] != bridge.hf_config.vision_end_token_id)
    input_mask = input_mask & (sample["input_ids"] != bridge.hf_config.vision_start_token_id)
    input_mask = input_mask & sample["input_ids"] != bridge.hf_config.audio_token_id
    input_mask = input_mask & (sample["input_ids"] != bridge.hf_config.audio_end_token_id)
    input_mask = input_mask & (sample["input_ids"] != bridge.hf_config.audio_start_token_id)
    input_mask = F.pad(input_mask[:, 1:], (0, 1, 0, 0), value=True)

    real_seq_length = sample["input_ids"].shape[-1]
    torch.distributed.barrier()
    seq_length_factor = args.tp
    if args.cp > 1:
        seq_length_factor *= args.cp * 2
    with torch.no_grad():
        fwd_bwd_function = get_forward_backward_func()

        seq_length = real_seq_length
        if real_seq_length % seq_length_factor != 0:
            seq_length = (
                (real_seq_length + seq_length_factor - 1)
                // seq_length_factor
                * seq_length_factor
            )
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
            if mpu.get_context_parallel_world_size() > 1:
                megatron_output = gather_output_from_cp(
                    megatron_output,
                    1,
                    mpu.get_context_parallel_world_size(),
                    mpu.get_context_parallel_group(),
                )
            if mpu.get_tensor_model_parallel_world_size() > 1:
                megatron_output = gather_from_tensor_model_parallel_region(
                    megatron_output
                )

            megatron_output = megatron_output[:, :real_seq_length, :]
            if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
                hf_output = torch.load("/tmp/hf_qwen3vl.pt", map_location="cpu").to(
                    megatron_output.device
                )

                print(f"======= cos_similarity without mask =======")
                cos_similarity(hf_output, megatron_output)
                # 去除 image-token-id 的再来做对比
                print(f"======= cos_similarity with mask =======")
                hf_output = hf_output[input_mask]
                megatron_output = megatron_output[input_mask]
                cos_similarity(hf_output, megatron_output)
                print(f"Finish Done")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
