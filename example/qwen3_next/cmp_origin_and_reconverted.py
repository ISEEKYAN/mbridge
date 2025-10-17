import torch
import os
from transformers import AutoModelForCausalLM, AutoConfig
import argparse

def compare_checkpoints(path1, path2, tolerance=1e-6):
    print(f"loading: {path1}")
    model1 = AutoModelForCausalLM.from_pretrained(path1,
                                                  dtype="auto",
                                                  device_map="auto")
    print(f"loading: {path2}")
    model2 = AutoModelForCausalLM.from_pretrained(path2,
                                                  dtype="auto",
                                                  device_map="auto")

    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    print(f"number of params: {len(state_dict1)} {len(state_dict2)}")

    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    if keys1 != keys2:
        print("❌ params name mismatch")
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        if only_in_1:
            print(f"params only in model1: {only_in_1}")
        if only_in_2:
            print(f"params only in model2: {only_in_2}")
        return False
    
    print("✅ params name match")

    all_match = True
    mismatch_count = 0

    error_key = []
    
    for key in state_dict1.keys():
        param1 = state_dict1[key]
        param2 = state_dict2[key]
        
        if param1.shape != param2.shape:
            print(f"❌ {key}: mismatch {param1.shape} vs {param2.shape}")
            all_match = False
            mismatch_count += 1
            error_key.append(key)
            continue

        if torch.allclose(param1, param2, atol=tolerance):
            print(f"✅ {key} shape {param1.shape} and weight is equal")
        else:
            # 计算差异统计
            diff = torch.abs(param1 - param2)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            print(f"❌ {key}: shape {param1.shape}, max diff: {max_diff:.6f}, avg diff: {mean_diff:.6f} {param1.sum()} {param2.sum()}")
            all_match = False
            mismatch_count += 1
            error_key.append(key)

    if all_match:
        print("match successfully")
    else:
        print(f"failed {mismatch_count} params {error_key=}")
    
    return all_match

def main():
    parser = argparse.ArgumentParser(description='compare')
    parser.add_argument('--path1', type=str, required=True, help='fisrt checkpoint')
    parser.add_argument('--path2', type=str, required=True, help='second checkpoint')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='')
    
    args = parser.parse_args()

    if not os.path.exists(args.path1):
        print(f"error: {args.path1}")
        return
    
    if not os.path.exists(args.path2):
        print(f"error: {args.path2}")
        return
    
    compare_checkpoints(args.path1, args.path2, args.tolerance)

if __name__ == "__main__":
    main()
