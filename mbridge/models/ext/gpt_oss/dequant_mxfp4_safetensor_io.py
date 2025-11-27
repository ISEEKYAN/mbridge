# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import math
import os
import warnings
from collections import defaultdict
from glob import glob
from typing import Generator

import torch
from safetensors import safe_open

from mbridge.core.safetensor_io import SafeTensorIO


class DequantMXFP4SafeTensorIO(SafeTensorIO):
    def __init__(self, hf_dir: str):
        super().__init__(hf_dir)

    def load_some_hf_weight(self, hf_weight_names: list[str]) -> dict:
        weight_to_file_map = self.index
        hf_dir = self.hf_dir
        ret = {}

        def is_mxfp4_weight(name):
            return "experts." in name and not name.endswith("bias")

        assert weight_to_file_map is not None, "index is not found"
        file_to_weight_map = defaultdict(set)
        for name in hf_weight_names:
            if isinstance(name, tuple):
                name = name[0]
            if is_mxfp4_weight(name):
                if f"{name}_blocks" in weight_to_file_map:
                    filename = weight_to_file_map[f"{name}_blocks"]
                else:
                    # load fp16
                    is_mxfp4_weight = lambda name: False
                    filename = weight_to_file_map[name]

            else:
                filename = weight_to_file_map[name]
            file_to_weight_map[filename].add(name)
        for filename, weight_names in file_to_weight_map.items():
            safetensor_file = os.path.join(hf_dir, filename)
            with safe_open(safetensor_file, framework="pt") as f:

                def get_another_tensor(name):
                    if weight_to_file_map[name] == filename:
                        return f.get_tensor(name)
                    else:
                        with safe_open(
                            os.path.join(hf_dir, weight_to_file_map[name]),
                            framework="pt",
                            device="cpu",
                        ) as f2:
                            return f2.get_tensor(name)

                for name in weight_names:
                    if is_mxfp4_weight(name):
                        blocks_name = f"{name}_blocks"
                        scale_name = f"{name}_scales"
                        blocks = get_another_tensor(blocks_name)
                        scales = get_another_tensor(scale_name)
                        scales = scales.to(torch.int32) - 127
                        weight = self._dequantize_mxfp4(blocks, scales)
                        if "gate_up_proj" in name or "down_proj" in name:
                            weight = weight.permute(
                                0, 2, 1
                            ).contiguous()  # make gate_up the last dimension
                    else:
                        weight = f.get_tensor(name)
                    ret[name] = weight
        return ret

    def _dequantize_mxfp4(
        self,
        blocks: torch.Tensor,
        scales: torch.Tensor,
        *,
        dtype: torch.dtype = torch.bfloat16,
        rows_per_chunk: int = 32768 * 1024,
    ) -> torch.Tensor:
        assert (
            blocks.shape[:-1] == scales.shape
        ), f"{blocks.shape=} does not match {scales.shape=}"
        FP4_VALUES = [
            +0.0,
            +0.5,
            +1.0,
            +1.5,
            +2.0,
            +3.0,
            +4.0,
            +6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
        lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

        *prefix_shape, G, B = blocks.shape
        rows_total = math.prod(prefix_shape) * G

        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)

            blk = blocks[r0:r1]
            exp = scales[r0:r1]

            # nibble indices -> int64
            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)

            sub = out[r0:r1]
            sub[:, 0::2] = lut[idx_lo]
            sub[:, 1::2] = lut[idx_hi]

            torch.ldexp(sub, exp, out=sub)
            del idx_lo, idx_hi, blk, exp

        return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
