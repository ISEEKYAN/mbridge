import json
import os
import warnings
from collections import defaultdict
from glob import glob
from typing import Generator

import torch
from transformers import AutoConfig
from safetensors import safe_open
from safetensors.torch import save_file

from mbridge.core.safetensor_io import SafeTensorIO


class Qwen3p5SafeTensorIO(SafeTensorIO):
    def __init__(self, hf_dir: str, ignore_mtp: bool = False):
        index_file = os.path.join(hf_dir, "model.safetensors.index.json")
        config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)

        self.index = {}
        self.origin_index = {}
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                origin_index = json.load(f)

                filtered_index = {}
                for key, value in origin_index["weight_map"].items():
                    if ignore_mtp and "mtp" in key:
                        continue
                    filtered_index[key] = value
                origin_index["weight_map"] = filtered_index

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
                            if ignore_mtp and "mtp" in key:
                                continue
                            self.index[key] = filename

        self.hf_dir = hf_dir
