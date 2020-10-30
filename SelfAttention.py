import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        
        assert self.embed_size % self.heads == 0, "Embedding size should be divisible by the number of heads."
        
        self.head_dim = self.embed_size // self.heads
        
        self.values_linear = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys_linear = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries_linear = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        self.fc_linear = nn.Linear(self.head_dim * self.heads, self.heads_dim * self.heads)
    
    def scaled_dot_product_attention(self, values: torch.Tensor, keys: torch.Tensor, queries: torch.Tensor,
                                     mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes 'Scaled Dot Product Attention'. SCDP(V, K, Q) = softmax((Q * K') / sqrt(d_k)) * V
        """
        # dim: n_batch * heads * q * k
        scores = (torch.matmul(queries, keys.transpose(-1, -2)) / (self.embed_size ** 0.5))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e20)
        
        attn_prob = F.softmax(scores, dim=-1)
        
        # values dim: n_batch * heads * len_values * head_dim
        # NOTE: len_values == len_keys. In both Encoder & Decoder part.
        # Output dim: n_batch * q * heads * head_dim
        return torch.matmul(attn_prob.transpose(1, 2), values)
    
    def forward(self, values: torch.Tensor, keys: torch.Tensor, queries: torch.Tensor, mask: torch.Tensor):
        """
        values, keys & queries are of dimension R^{batch_size * words * embed_dim}
        """
        batch_size, embed_dim = values.shape[0], values.shape[2]
        len_values, len_keys, len_queries = values.shape[1], keys.shape[1], queries.shape[1]
        
        values = values.reshape(batch_size, len_values, self.heads, self.head_dim)
        keys = keys.reshape(batch_size, len_keys, self.heads, self.head_dim)
        queries = queries.reshape(batch_size, len_queries, self.heads, self.head_dim)
        
        # dimension: n_batch * heads * len * head_dim
        values, keys, queries = [f(x).transpose(1, 2) for f, x in
                                 zip([self.values_linear, self.keys_linear, self.queries_linear],
                                     [values, keys, queries])]
        
        # dim: batch_size * q * heads * head_dim
        attn_scores = self.scaled_dot_product_attention(values, keys, queries, mask)
        
        # concatenation step by flattening out last two dimensions
        attn_scores = attn_scores.reshape(batch_size, len_queries, self.heads * self.head_dim)
        
        # final linear layer
        # dim: batch_size * q * embed_size
        return self.fc_linear(attn_scores)
