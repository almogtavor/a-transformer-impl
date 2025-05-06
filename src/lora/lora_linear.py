import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear, used only when LoRA is desired (e.g. during fine-tuning).
    Keeps the original weight frozen and adds a trainable low-rank adapter.
    """
    def __init__(self, in_features, out_features, r=4, alpha=1.0, dropout=0.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha

        # Frozen base weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Optional trainable bias
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=bias) if bias else None

        # LoRA trainable adapters
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)

        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        lora = self.dropout(x) @ self.A.T
        lora = lora @ self.B.T
        return base + self.scaling * lora