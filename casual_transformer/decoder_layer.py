import torch.nn as nn
from casual_transformer.multi_head_attention import MultiHeadAttention
from casual_transformer.position_wise_feed_forward_nn import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, use_lora=False):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, use_lora=use_lora)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, use_lora=use_lora)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, no_peak_mask):
        """
        x (torch.Tensor): input tensor of shape (batch_size, target_seq_length, d_model).
        enc_output (torch.Tensor): the encoder output, tensor of shape (batch_size, source_seq_length, d_model).
        no_peak_mask (torch.Tensor): the mask for the decoder, tensor of shape (1, target_seq_length, target_seq_length).

        """
        self_attn_output = self.self_attn(x, x, x, mask=no_peak_mask)
        residual_output = x + self.dropout(self_attn_output)
        x = self.norm1(residual_output)

        cross_attn_output = self.cross_attn(x, enc_output, enc_output)
        residual_output = x + self.dropout(cross_attn_output)
        x = self.norm2(residual_output)

        ff_output = self.feed_forward(x)
        residual_output = x + self.dropout(ff_output)
        x = self.norm3(residual_output)
        return x