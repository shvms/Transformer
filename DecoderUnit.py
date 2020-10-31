import torch
import torch.nn as nn

from SelfAttention import SelfAttention
from EncoderUnit import EncoderUnit


class DecoderUnit(nn.Module):
    def __init__(self, embed_size: int, heads: int, forward_expansion: int,
                 dropout: float = 0.5,
                 device: torch.device = torch.device('cpu')):
        super(DecoderUnit, self).__init__()
        
        self.masked_attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.encoder_unit = EncoderUnit(embed_size, heads, forward_expansion, dropout)
        self.dropout = nn.Dropout(dropout)
        self.device = device
    
    def forward(self, x: torch.Tensor, value: torch.Tensor, key: torch.Tensor, tgt_mask: torch.Tensor,
                src_mask: torch.Tensor = None):
        # masked multi-head attention
        scores = self.masked_attention(x, x, x, mask=tgt_mask)
        
        # add & norm
        query = self.dropout(self.norm(scores + x))
        
        # encoder unit: multi-head attention -> add & norm -> feed-forward -> add & norm
        return self.encoder_unit(value, key, query, src_mask)
