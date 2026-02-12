import torch


def cos_similarity(a, b):
    print(f"a {a.shape} b {b.shape}")
    a = a.to(b.device)
    a = a.float()
    # a = a / a.norm(dim=-1, keepdim=True)
    a = torch.exp(a)
    a = a / a.norm(dim=-1, keepdim=True)
    """
    a = (a - a.mean(dim=-1, keepdim=True)) 
    a = a / a.norm(dim=-1, keepdim=True)
    """
    b = b.float()
    # b =  b / b.norm(dim=-1, keepdim=True)
    b = torch.exp(b)
    b = b / b.norm(dim=-1, keepdim=True)
    """
    b = (b - b.mean(dim=-1, keepdim=True)) 
    b =  b / b.norm(dim=-1, keepdim=True)
    """
    sim = (a * b).sum(dim=-1)
    print(
        f"hf vs megatron cos_similarity min: {sim.min()}; max: {sim.max()}; mean: {sim.mean()}"
    )


path1 = "qwen3p5_save/hf_qwen3_5.pt"

path2_list = [
    # "qwen3p5_save/mlm_tp1_pp1_cp1_ep1.pt",
    # "qwen3p5_save/mlm_tp2_pp1_cp1_ep1.pt",
    # "qwen3p5_save/mlm_tp2_pp1_cp1_ep4.pt",
    # "qwen3p5_save/mlm_tp2_pp1_cp2_ep4.pt",
    "qwen3p5_save/mlm_tp2_pp2_cp2_ep4.pt",
    "qwen3p5_save/mlm_tp4_pp2_cp1_ep4.pt",
    "qwen3p5_save/mlm_tp2_pp2_cp1_ep2.pt",
]

a = torch.load(path1)
for path2 in path2_list:
    print(f"load from {path1=} {path2=}")
    b = torch.load(path2)

    cos = cos_similarity(a, b)
    print(f"{cos=} {a.sum()} {b.sum()} {a.dtype} {b.dtype}")
