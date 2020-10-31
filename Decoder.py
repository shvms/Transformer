import torch
import torch.nn as nn

from clone import get_clones
from DecoderUnit import DecoderUnit

class Decoder(nn.Module):
    def __init__(
        self,
        tgt_vocab_size: int,
        embed_size: int,
        heads: int,
        n_layers: int,
        max_len: int,
        device: torch.device,
        forward_expansion: int,
        dropout: float = 0.5
    ):
        super(Decoder, self).__init__()

        self.tgt_vocab_size = tgt_vocab_size
        self.embed_size = embed_size
        self.heads = heads
        self.n_layers = n_layers
        self.max_len = max_len
        self.device = device
        self.forward_expansion = forward_expansion
        self.dropout = dropout
        
        self.word_embeddings = nn.Embedding(tgt_vocab_size, embed_size)
        self.positional_embeddings = nn.Embedding(max_len, embed_size)
        
        self.decoder_units = get_clones(DecoderUnit(embed_size, heads, forward_expansion, dropout, device))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, tgt_mask: torch.Tensor, src_mask: torch.Tensor = None):
        N, sent_length = x.shape
        positions = torch.arange(0, sent_length).expand(N, sent_length).to(self.device)
        
        x = self.dropout(self.word_embeddings(x) + self.positional_embeddings(positions))
        
        for decoder_unit in self.decoder_units:
            x = decoder_unit(x, enc_out, enc_out, tgt_mask, src_mask)
        
        return x
