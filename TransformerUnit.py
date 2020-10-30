import torch
import torch.nn as nn

from SelfAttention import SelfAttention


class TransformerUnit(nn.Module):
    def __init__(self, embed_size: int, heads: int, forward_expansion: int, dropout: float = 0.5):
        super(TransformerUnit, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.dropout = dropout
        self.forward_expansion = forward_expansion
        
        self.self_attention = SelfAttention(embed_size, heads)  # dim: batch_size * q * embed_size
        self.norm1 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, values: torch.Tensor, keys: torch.Tensor, queries: torch.Tensor, mask: torch.Tensor = None):
        attn_scores = self.self_attention(values, keys, queries, mask)
        
        normalised_score = self.dropout(self.norm1(attn_scores + queries))
        feed_forward_score = self.feed_forward(normalised_score)
        
        return self.dropout(self.norm2(feed_forward_score + normalised_score))
