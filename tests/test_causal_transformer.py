import pytest
import torch
import torch.nn as nn

from casual_transformer.casual_transformer import CausalTransformer
from casual_transformer.decoder_layer import DecoderLayer
from casual_transformer.encoder_layer import EncoderLayer
from casual_transformer.positional_encoding import PositionalEncoding


# Test Case 1: Verify model initialization
def test_causal_transformer_initialization():
    # Define some parameters
    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 128
    dropout = 0.1

    # Initialize the model
    model = CausalTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length,
                              dropout)

    # Check that the model's components are initialized correctly
    assert isinstance(model.encoder_embedding, nn.Embedding)
    assert isinstance(model.decoder_embedding, nn.Embedding)
    assert isinstance(model.positional_encoding,
                      PositionalEncoding)  # Assuming PositionalEncoding class is implemented elsewhere
    assert len(model.encoder_layers) == num_layers
    assert len(model.decoder_layers) == num_layers
    assert isinstance(model.fc, nn.Linear)
    assert isinstance(model.dropout, nn.Dropout)

    # Check that the layers in the model are of the expected types
    assert isinstance(model.encoder_layers[0], EncoderLayer)
    assert isinstance(model.decoder_layers[0], DecoderLayer)


# Test Case 2: Verify mask generation
def test_generate_mask():
    # Define some parameters
    batch_size = 2
    tgt_seq_length = 5
    tgt = torch.randint(0, 100, (batch_size, tgt_seq_length))  # Random target sequence

    model = CausalTransformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_length=128,
        dropout=0.1
    )
    mask = model.generate_mask(tgt)

    # Check the shape of the mask
    assert mask.shape == (1, tgt_seq_length,
                          tgt_seq_length), f"Expected mask shape (1, {tgt_seq_length}, {tgt_seq_length}), but got {mask.shape}"

    # Convert to numpy for inspection
    mask_data = mask.squeeze(0).cpu().numpy()

    # Check that all values above the diagonal are 0 and on/below are 1
    for i in range(tgt_seq_length):
        for j in range(tgt_seq_length):
            if j > i:
                assert mask_data[i, j] == 0, f"Mask at ({i}, {j}) should be 0"
            else:
                assert mask_data[i, j] == 1, f"Mask at ({i}, {j}) should be 1"


# Test Case 3: Verify forward pass
def test_forward_pass():
    # Define some parameters
    batch_size = 2
    src_seq_length = 10
    tgt_seq_length = 5

    src = torch.randint(0, 100, (batch_size, src_seq_length))  # Random source sequence
    tgt = torch.randint(0, 100, (batch_size, tgt_seq_length))  # Random target sequence

    model = CausalTransformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_length=128,
        dropout=0.1
    )

    output = model(src, tgt)  # Perform a forward pass

    # Check that the output has the expected shape
    assert output.shape == (batch_size, tgt_seq_length, 100)  # (batch_size, tgt_seq_len, vocab_size)


# Test Case 4: Ensure model can handle different sequence lengths
@pytest.mark.parametrize("seq_length", [5, 10, 50, 100])
def test_variable_sequence_length(seq_length):
    batch_size = 2
    src = torch.randint(0, 100, (batch_size, seq_length))  # Random source sequence with variable length
    tgt = torch.randint(0, 100, (batch_size, seq_length))  # Random target sequence with variable length

    model = CausalTransformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_length=128,
        dropout=0.1
    )

    output = model(src, tgt)  # Perform a forward pass

    # Check that the output shape is correct regardless of sequence length
    assert output.shape == (batch_size, seq_length, 100)  # (batch_size, tgt_seq_len, vocab_size)
