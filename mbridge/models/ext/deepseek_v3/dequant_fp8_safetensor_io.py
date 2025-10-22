import json
import os
import warnings
from collections import defaultdict
from glob import glob
from typing import Generator

import torch
from safetensors import safe_open

from mbridge.core.safetensor_io import SafeTensorIO

from .kernel import weight_dequant


class DequantFP8SafeTensorIO(SafeTensorIO):
    def __init__(self, hf_dir: str):
        super().__init__(hf_dir)

    def load_some_hf_weight(self, hf_weight_names: list[str]) -> dict:
        # set default dtype to bfloat16
        torch.set_default_dtype(torch.bfloat16)
        weight_to_file_map = self.index
        hf_dir = self.hf_dir
        ret = {}

        assert weight_to_file_map is not None, "index is not found"
        file_to_weight_map = defaultdict(list)
        for name in hf_weight_names:
            filename = weight_to_file_map[name]
            file_to_weight_map[filename].append(name)
        for filename, weight_names in file_to_weight_map.items():
            safetensor_file = os.path.join(hf_dir, filename)
            with safe_open(safetensor_file, framework="pt", device="cuda") as f:
                for name in weight_names:
                    weight = f.get_tensor(name)
                    scale_inv_name = f"{name}_scale_inv"
                    if (
                        weight.element_size() == 1
                        and scale_inv_name in weight_to_file_map
                    ):  # FP8 weight
                        try:
                            if weight_to_file_map[scale_inv_name] == filename:
                                scale_inv = f.get_tensor(scale_inv_name)
                            else:
                                with safe_open(
                                    os.path.join(
                                        hf_dir, weight_to_file_map[scale_inv_name]
                                    ),
                                    framework="pt",
                                    device="cuda",
                                ) as f2:
                                    scale_inv = f2.get_tensor(scale_inv_name)
                            ret[name] = weight_dequant(weight, scale_inv)
                        except KeyError:
                            print(
                                f"Warning: Missing scale_inv tensor for {name}, skipping conversion"
                            )
                            ret[name] = weight
                    else:
                        ret[name] = weight
        # set default dtype back to float32
        torch.set_default_dtype(torch.float32)
        return ret

    def get_keys_maps_to_save(self) -> dict:
        filename_to_keys_map = defaultdict(set)
        for key, filename in self.index.items():
            if key.endswith("_scale_inv"):
                continue
            filename_to_keys_map[filename].add(key)
        return filename_to_keys_map
    
    def save_index(self, new_hf_dir: str):
        if self.origin_index:
            weight_map = {}
            for key, filename in self.origin_index['weight_map'].items():
                if key.endswith("_scale_inv"):
                    continue
                weight_map[key] = filename
            self.origin_index['weight_map'] = weight_map
            with open(
                os.path.join(new_hf_dir, "model.safetensors.index.json"), "w"
            ) as f:
                json.dump(self.origin_index, f)
        else:
            warnings.warn("No index file found, saving index file failed")
        return