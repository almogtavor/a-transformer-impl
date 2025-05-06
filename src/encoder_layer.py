import torch.nn as nn
from src.multi_head_attention import MultiHeadAttention
from src.position_wise_feed_forward_nn import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        x (torch.Tensor): input tensor of shape (batch_size, seq_length, d_model).
        """
        ## Question 4 - your code here
        # Apply self-attention mechanism, add the residual connection and apply dropout
        attn_output = self.self_attn(x, x, x)  # Self-attention: Q = K = V = x
        # Add residual, apply dropout, then layer norm
        residual_output = x + self.dropout(attn_output)
        x = self.norm1(residual_output)

        # Apply the feed-forward layer, add the residual connection and apply dropout
        ff_output = self.feed_forward(x)
        residual_output = x + self.dropout(ff_output)
        x = self.norm2(residual_output)
        return x
        ## End of your code