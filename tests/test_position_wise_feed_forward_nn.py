import torch
from torch import nn

from casual_transformer.position_wise_feed_forward_nn import PositionWiseFeedForward


def test_positionwise_feedforward_shapes():
    batch_size = 2
    seq_len = 5
    d_model = 16
    d_ff = 64

    x = torch.randn(batch_size, seq_len, d_model)
    ff = PositionWiseFeedForward(d_model, d_ff)
    output = ff(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
