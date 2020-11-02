import torch
import torch.nn as nn

from .clone import get_clones
from .EncoderUnit import EncoderUnit


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        embed_size: int,
        heads: int,
        n_layers: int,
        max_len: int,
        device: torch.device,
        forward_expansion: int,
        dropout: float = 0.5,
    ):
        super(Encoder, self).__init__()
        
        self.src_vocab_size = src_vocab_size
        self.embed_size = embed_size
        self.heads = heads
        self.n_layers = n_layers
        self.max_len = max_len
        self.device = device
        self.forward_expansion = forward_expansion
        self.dropout = dropout
        
        self.word_embeddings = nn.Embedding(self.src_vocab_size, self.embed_size)
        self.positional_embeddings = nn.Embedding(self.max_len, self.embed_size)
        
        self.encoder_units = get_clones(
            EncoderUnit(self.embed_size, self.heads, self.forward_expansion, self.dropout),
            self.n_layers
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        N, sent_length = x.shape
        positions = torch.arange(0, sent_length).expand(N, sent_length).to(self.device)
        x = self.dropout(self.word_embeddings(x) + self.positional_embeddings(positions))
        
        for encoder_unit in self.encoder_units:
            x = encoder_unit(values=x, keys=x, queries=x, mask=mask)
        
        return x
