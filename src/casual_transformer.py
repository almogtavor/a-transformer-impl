import torch
import torch.nn as nn

from src.decoder_layer import DecoderLayer
from src.encoder_layer import EncoderLayer
from src.multi_head_attention import MultiHeadAttention
from src.position_wise_feed_forward_nn import PositionWiseFeedForward
from src.positional_encoding import PositionalEncoding


class CausalTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, use_lora=False):
        super(CausalTransformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, use_lora=use_lora)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, use_lora=use_lora)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, tgt):
        """
        tgt (torch.Tensor): the target sequence, tensor of shape (batch_size, target_seq_length)

        Returns:
        torch.Tensor: a no peak map for the target_data, of shape
                      (1, seq_length, seq_length).
        """
        ## Question 6 - your code here
        target_seq_len = tgt.size(1)  # Length of the target sequence
        mask = torch.ones(target_seq_len, target_seq_len)
        mask = mask - torch.triu(mask, diagonal=1)  # Set the upper triangle to 0 to block future tokens
        return mask.unsqueeze(0)  # Add batch dimension (1, target_seq_len, target_seq_len)
        ## End of your code

    def forward(self, src, tgt):
        np_mask = self.generate_mask(tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, np_mask)

        output = self.fc(dec_output)
        return output