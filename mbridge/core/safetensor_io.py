import json
import logging
import os
import warnings
from collections import defaultdict
from glob import glob
from typing import Generator

import torch
from transformers import AutoConfig
from safetensors import safe_open
from safetensors.torch import save_file


logger = logging.getLogger(__name__)


class SafeTensorIO:
    def __init__(self, hf_dir: str):
        index_file = os.path.join(hf_dir, "model.safetensors.index.json")
        config = AutoConfig.from_pretrained(hf_dir)

        self.index = {}
        self.origin_index = {}
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                origin_index = json.load(f)
                self.index = origin_index["weight_map"]
                if getattr(config, "tie_word_embeddings", False):
                    if "lm_head.weight" in self.index.keys():
                        self.index.pop("lm_head.weight")
                self.origin_index = origin_index
        else:
            src_files = glob(os.path.join(hf_dir, "*.safetensors"))
            if len(src_files) == 1:
                for file in src_files:
                    with safe_open(file, framework="pt", device="cpu") as f:
                        filename = os.path.basename(file)
                        for key in f.keys():
                            self.index[key] = filename

        self.hf_dir = hf_dir

    # Some models have undergone structural changes across different versions of transformers,
    # which may result in differences in key names
    # for example: qwen2.5vl's structural change at transformers>=4.52.0
    # Therefore, it is necessary to create two name mappings here.
    def _mapping_weight_names_old2new(
        self,
        hf_weight_names: list[str],
    ) -> tuple[list[str], dict[str, str]]:
        # new_hf_weight_names -> old_hf_weight_names
        mapping_hf_weight_names = {k: k for k in hf_weight_names}
        return hf_weight_names, mapping_hf_weight_names

    def _mapping_weight_names_new2old(
        self,
        hf_weight_names: list[str],
    ) -> tuple[list[str], dict[str, str]]:
        # new_hf_weight_names -> old_hf_weight_names
        mapping_hf_weight_names = {k: k for k in hf_weight_names}
        return hf_weight_names, mapping_hf_weight_names

    def load_some_hf_weight(self, hf_weight_names: list[str]) -> dict:
        hf_weight_names, mapping_hf_weight_names = self._mapping_weight_names_old2new(
            hf_weight_names
        )
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
            return {mapping_hf_weight_names[k]: v for k, v in ret.items()}
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
            return {mapping_hf_weight_names[k]: v for k, v in ret.items()}
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

    def get_keys_maps_to_save(self) -> dict:
        filename_to_keys_map = defaultdict(set)
        for key, filename in self.index.items():
            filename_to_keys_map[filename].add(key)
        return filename_to_keys_map

    def save_hf_weight(
        self,
        per_tensor_generator: Generator[tuple[str, torch.Tensor], None, None],
        new_hf_dir: str,
        hf_shared_weight_keys: list[str],
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

        filename_to_keys_map = self.get_keys_maps_to_save()
        states = {}
        saved_files = []
        collected_weights_count = 0

        for hf_weight_name, tensor in per_tensor_generator:
            states[hf_weight_name] = tensor.cpu()
            collected_weights_count += 1

            # Check if any file can be saved now
            for filename, keys_for_file in filename_to_keys_map.items():
                if keys_for_file.issubset(states.keys()):
                    to_save = {k: states[k] for k in keys_for_file}
                    safetensor_file = os.path.join(new_hf_dir, filename)
                    logger.info(f"[SafeTensorIO] ✓ All {len(keys_for_file)} weights collected for '{filename}', saving...")
                    save_file(to_save, safetensor_file)
                    saved_files.append(filename)

                    # Remove saved weights from states
                    for k in keys_for_file:
                        del states[k]
                    logger.info(f"[SafeTensorIO] ✓ Saved '{filename}', remaining weights in buffer: {len(states)}")

        # Save remaining weights that couldn't form a complete file
        # (e.g. when some weights like MTP weights are not in the generator)
        if states:
            remaining_file_map = defaultdict(dict)
            unknown_weights = {}
            for weight_name, tensor in states.items():
                if weight_name in self.index:
                    expected_file = self.index[weight_name]
                    remaining_file_map[expected_file][weight_name] = tensor
                else:
                    unknown_weights[weight_name] = tensor

            for filename, to_save in remaining_file_map.items():
                safetensor_file = os.path.join(new_hf_dir, filename)
                expected_weights = filename_to_keys_map[filename]
                missing_weights = expected_weights - set(to_save.keys())
                if os.path.exists(safetensor_file):
                    # File was already saved (fully complete), merge remaining weights into it
                    with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                        for k in f.keys():
                            if k not in to_save:
                                to_save[k] = f.get_tensor(k)
                logger.info(
                    f"[SafeTensorIO] ✓ Saving partial file '{filename}' with "
                    f"{len(to_save)}/{len(expected_weights)} weights "
                    f"(missing {len(missing_weights)}: {list(missing_weights)[:5]}{'...' if len(missing_weights) > 5 else ''})"
                )
                save_file(to_save, safetensor_file)
                saved_files.append(filename)

            if unknown_weights:
                for weight_name in unknown_weights:
                    logger.warning(f"[SafeTensorIO] ⚠️  '{weight_name}' is NOT in the index file, skipping!")

            # Clear saved weights from states
            unsaved = set(unknown_weights.keys())
            states = {k: v for k, v in states.items() if k in unsaved}

        logger.info(f"\n[SafeTensorIO] === Save Summary ===")
        logger.info(f"[SafeTensorIO] Total weights collected: {collected_weights_count}")
        logger.info(f"[SafeTensorIO] Files saved: {len(saved_files)}/{len(filename_to_keys_map)}")
        logger.info(f"[SafeTensorIO] Weights remaining in buffer: {len(states)}")

        if states:
            warnings.warn(
                f"Some weights are not saved: {states.keys()} {hf_shared_weight_keys=}"
            )
        return

    def save_tmp_weight(
        self,
        hf_weight_name: str,
        tensor: torch.tensor,
        new_hf_dir: str,
    ):
        assert self.index, "index file is required for memory efficient saving"
        tmp_filename = f"{new_hf_dir}/{hf_weight_name}.safetensors"
        save_file({hf_weight_name: tensor}, tmp_filename)

    def save_tmp_hf_weights(
        self,
        per_tensor_generator: Generator[tuple[str, torch.Tensor], None, None],
        new_hf_dir: str,
    ):
        assert self.index, "index file is required for memory efficient saving"
        for hf_weight_name, tensor in per_tensor_generator:
            tmp_filename = f"{new_hf_dir}/{hf_weight_name}.safetensors"
            save_file({hf_weight_name: tensor}, tmp_filename)

    def save_hf_weight_merge(
        self,
        new_hf_dir: str,
        rank: int = 0,
        world_size: int = 1,
    ):
        assert self.index, "index file is required for memory efficient saving"

        filename_to_keys_map = defaultdict(set)
        for key, filename in self.index.items():
            filename_to_keys_map[filename].add(key)

        filename_list = sorted(list(filename_to_keys_map.keys()))
        if world_size > 1:
            num_files = len(filename_list)
            num_files_rank = (num_files + world_size - 1) // world_size
            begin_idx = min(num_files, rank * num_files_rank)
            end_idx = min(num_files, (rank + 1) * num_files_rank)
            filename_list = filename_list[begin_idx:end_idx]

        for filename in filename_list:
            keys_for_file = filename_to_keys_map[filename]
            states = {}
            old_keys_for_file, _ = self._mapping_weight_names_new2old(keys_for_file)
            for old_key, key in zip(old_keys_for_file, keys_for_file):
                tmp_filename = f"{new_hf_dir}/{old_key}.safetensors"
                if not os.path.exists(tmp_filename):
                    logger.warning(f"[SafeTensorIO] ERROR: File '{tmp_filename}' not exists，skipping!")
                    continue
                with safe_open(tmp_filename, framework="pt", device="cpu") as f:
                    tensor = f.get_tensor(old_key)
                    states[key] = tensor.clone()
                os.remove(tmp_filename)

            if states:
                save_file(states, os.path.join(new_hf_dir, filename))
                logger.info(f"[SafeTensorIO] Successfully saved {filename} with {len(states)} tensors")
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

        self.save_tmp_hf_weights(per_tensor_generator, new_hf_dir)
        self.save_hf_weight_merge(new_hf_dir)
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
