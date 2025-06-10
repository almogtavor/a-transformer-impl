import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math

from casual_transformer.casual_transformer import CausalTransformer

torch.manual_seed(33)

my_dim = 20
x = torch.randint(-10,10,(my_dim,)) # Any Random Vector or Matrix
my_relu = nn.ReLU()
my_relu(x)
my_linear = nn.Linear(my_dim, my_dim*3) # Linear Model
y = torch.randn(my_dim, my_dim) # Any Random Vector or Matrix
output = my_linear(y)
print(output.size())


def train():
    # === Logging Setup ===
    os.makedirs("training_results", exist_ok=True)
    count_file = "training_results/training_log.txt"

    # Get and update training number
    if os.path.exists(count_file):
        with open(count_file, "r") as f:
            count = int(f.read().strip())
    else:
        count = 0

    count += 1
    with open(count_file, "w") as f:
        f.write(str(count))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"training_results/{timestamp}_train{count}.txt"

    log_lines = [f"This is training number {count}\n"]

    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1

    transformer = CausalTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff,
                                    max_seq_length, dropout, use_lora=True)

    # Generate random sample data
    src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    for epoch in range(10):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        log_line = f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}"
        print(log_line)
        log_lines.append(log_line + "\n")
    # Write all logs to file
    with open(log_filename, "w") as f:
        f.writelines(log_lines)

if __name__ == '__main__':
    train()