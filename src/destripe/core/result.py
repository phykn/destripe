from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StripeResult:
    clean: torch.Tensor
    components: tuple[torch.Tensor, ...]
