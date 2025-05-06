import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math

torch.manual_seed(33)

my_dim = 20
x = torch.randint(-10, 10, (my_dim,))  # Any Random Vector or Matrix
my_relu = nn.ReLU()
my_relu(x)
my_linear = nn.Linear(my_dim, my_dim * 3)  # Linear Model
y = torch.randn(my_dim, my_dim)  # Any Random Vector or Matrix
output = my_linear(y)
print(output.size())


# Multi-Head Attention (MHA)
# typical MHA tensor have the shape of [batch_size, num_heads, seq_len, head_dim]
# so we'll take K.transpose(-2,-1) (equivalent to K.transpose(2,3), i.e. seq_len, head_dim)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "error"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Parameter(torch.Tensor(num_heads, d_model, self.d_k))
        self.W_k = nn.Parameter(torch.Tensor(num_heads, d_model, self.d_k))
        self.W_v = nn.Parameter(torch.Tensor(num_heads, d_model, self.d_k))
        self.W_o = nn.Linear(d_model, d_model)

        self.init_weights()

    def init_weights(self):
        """Initialize weight matrices with Xavier initialization."""
        for param in [self.W_q, self.W_k, self.W_v]:
            nn.init.xavier_uniform_(param)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_length_q,  d_k).
        K (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_length_v, d_k).
        V (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_length_v, d_k).
        mask (torch.Tensor): Mask tensor of shape (1, seq_length_q, seq_length_v), where 0
                                       indicates masked positions. Default is None.

        Returns:
        torch.Tensor: The output tensor after applying attention, of shape
                      (batch_size, num_heads, seq_length_q, d_k).
        """
        ## Question 1 - implement this function
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        # for each (batch, head, query), we get a probability distribution over all key positions
        # there are seq_length_k (or seq_length_v) keys, and this is the last dimention of the attn_scores tensor
        attn_weights = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, V)
        ### End of your code

    def forward(self, Q, K, V, mask=None):
        batch_size, seq_len, _ = Q.shape

        # Project each head separately using Einstein summation
        # Q_heads, K_heads, and V_heads each have separate projections per head.
        Q_heads = torch.einsum("bnd,hde->bhne", Q, self.W_q)
        K_heads = torch.einsum("bnd,hde->bhne", K, self.W_k)
        V_heads = torch.einsum("bnd,hde->bhne", V, self.W_v)

        attn_output = self.scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)
        return output
