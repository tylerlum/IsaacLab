from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class AverageMeter(nn.Module):
    def __init__(self, in_shape: int = 1, max_size: int = 1000) -> None:
        super().__init__()
        self.max_size = max_size

        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values: torch.Tensor) -> None:
        assert len(values.shape) == 1, f"values.shape: {values.shape}"
        size = values.size()[0]
        if size == 0:
            return

        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self) -> None:
        self.current_size = 0
        self.mean.fill_(0.0)

    def __len__(self) -> int:
        return self.current_size

    def get_mean(self) -> np.ndarray:
        return self.mean.squeeze(0).cpu().numpy()
