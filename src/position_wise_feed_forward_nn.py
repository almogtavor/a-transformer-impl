import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.relu = nn.ReLU()
        ## Question 2a - fill in the dimensions
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        x (torch.Tensor): input tensor of shape (batch_size, seq_length, d_model).
        """
        ## Question 2b - your code here
        w1_output = self.w1(x)
        relu_output = self.relu(w1_output)
        return self.w2(relu_output)
        ## End of your code