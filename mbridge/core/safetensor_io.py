import json
import os
import warnings
from collections import defaultdict
from glob import glob
from typing import Generator

import torch
from safetensors import safe_open
from safetensors.torch import save_file


class SafeTensorIO:
    def __init__(self, hf_dir: str):
        index_file = os.path.join(hf_dir, "model.safetensors.index.json")

        self.index = {}
        self.origin_index = {}
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                origin_index = json.load(f)
                self.index = origin_index["weight_map"]
                self.origin_index = origin_index

        self.hf_dir = hf_dir

    def load_some_hf_weight(self, hf_weight_names: list[str]) -> dict:
        index = self.index
        hf_dir = self.hf_dir
        ret = {}

        if index:
            file_to_weight_map = defaultdict(list)
            for name in hf_weight_names:
                filename = index[name]
                file_to_weight_map[filename].append(name)
            for filename, weight_names in file_to_weight_map.items():
                safetensor_file = os.path.join(hf_dir, filename)
                with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                    for name in weight_names:
                        ret[name] = f.get_tensor(name)
            return ret
        # Search all safetensors files
        safetensor_files = glob(os.path.join(hf_dir, "*.safetensors"))
        # If there are safetensors files
        if safetensor_files:
            # Iterate through each safetensors file
            for safetensor_file in safetensor_files:
                with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                    to_load = set(hf_weight_names) & set(f.keys())
                    if to_load:
                        for name in to_load:
                            ret[name] = f.get_tensor(name)
                            # print(f"{name} {ret[name].shape}")
            if len(ret) != len(hf_weight_names):
                raise ValueError(
                    f"Weights {set(hf_weight_names)-set(ret.keys())} not found in safetensors files in {hf_dir}"
                )
            return ret
        if len(safetensor_files) == 0:
            if glob(os.path.join(hf_dir, "pytorch_model-*.bin")).__len__() > 0:
                raise NotImplementedError(
                    "Not implemented for deprecated pytorch_model-*.bin format huggingface weights, please use safetensors format"
                )
        raise ValueError(
            f"Weights {hf_weight_names} not found in safetensors files in {hf_dir}"
        )

    def load_one_hf_weight(self, hf_weight_name: str) -> torch.Tensor:
        return self.load_some_hf_weight([hf_weight_name])[hf_weight_name]

    def load_hf_weight_names(self) -> list[str]:
        if self.index:
            return list(self.index.keys())
        else:
            safetensor_files = glob(os.path.join(self.hf_dir, "*.safetensors"))
            ret = []
            for file in safetensor_files:
                with safe_open(file, framework="pt", device="cpu") as f:
                    ret.extend(f.keys())
            return ret

    def save_hf_weight(
        self,
        per_tensor_generator: Generator[tuple[str, torch.Tensor], None, None],
        new_hf_dir: str,
    ):
        """
        This function is used to save weights to a safetensors file.
        It will save weights to a single safetensors file if the model is small.
        It will save weights to multiple safetensors files if the model is large.
        """
        if not self.index:
            # for small model, we save all weights to a single safetensors file
            filename = f"model.safetensors"
            safetensor_file = os.path.join(new_hf_dir, filename)
            states = {}
            for hf_weight_name, tensor in per_tensor_generator:
                states[hf_weight_name] = tensor
            save_file(states, safetensor_file)
            return

        filename_to_keys_map = defaultdict(set)
        for key, filename in self.index.items():
            filename_to_keys_map[filename].add(key)
        states = {}
        for hf_weight_name, tensor in per_tensor_generator:
            states[hf_weight_name] = tensor.cpu()
            for filename, keys_for_file in filename_to_keys_map.items():
                if keys_for_file.issubset(states.keys()):
                    to_save = {k: states[k] for k in keys_for_file}
                    safetensor_file = os.path.join(new_hf_dir, filename)
                    save_file(to_save, safetensor_file)
                    for k in keys_for_file:
                        del states[k]
        assert len(states) == 0, f"Some weights are not saved: {states.keys()}"
        return

    def save_hf_weight_memory_efficient(
        self,
        per_tensor_generator: Generator[tuple[str, torch.Tensor], None, None],
        new_hf_dir: str,
    ):
        """
        This function is used to save weights in a memory efficient way.
        It will save weights to temporary files and then merge them into the final safetensors file.
        It is useful for large models that have many weights.
        """
        assert self.index, "index file is required for memory efficient saving"

        filename_to_keys_map = defaultdict(set)
        for key, filename in self.index.items():
            filename_to_keys_map[filename].add(key)
        for hf_weight_name, tensor in per_tensor_generator:
            tmp_filename = f"{new_hf_dir}/{hf_weight_name}.safetensors"
            save_file({hf_weight_name: tensor}, tmp_filename)
        for filename, keys_for_file in filename_to_keys_map.items():
            states = {}
            for key in keys_for_file:
                tmp_filename = f"{new_hf_dir}/{key}.safetensors"
                with safe_open(tmp_filename, framework="pt", device="cpu") as f:
                    states[key] = f.get_tensor(key)
                    os.remove(tmp_filename)
            save_file(states, os.path.join(new_hf_dir, filename))
        return

    def save_index(self, new_hf_dir: str):
        if self.origin_index:
            with open(
                os.path.join(new_hf_dir, "model.safetensors.index.json"), "w"
            ) as f:
                json.dump(self.origin_index, f)
        else:
            warnings.warn("No index file found, saving index file failed")
        return
