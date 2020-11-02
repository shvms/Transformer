import torch
import torch.nn as nn

from Encoder import Encoder
from Decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        embed_size: int = 256,
        heads: int = 8,
        n_layers: int = 6,
        max_len: int = 100,
        forward_expansion: int = 4,
        dropout: float = 0,
        device: torch.device = torch.device('cpu')
    ):
        super(Transformer, self).__init__()
        
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device
        
        self.encoder = Encoder(src_vocab_size, embed_size, heads, n_layers, max_len, device, forward_expansion, dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_size, heads, n_layers, max_len, device, forward_expansion, dropout)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        src_mask = self.get_src_mask(src)
        tgt_mask = self.get_tgt_mask(tgt)
        
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        return self.softmax(dec_out)
    
    def get_src_mask(self, src: torch.Tensor):
        # dim: N * 1 * 1 * src_len
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)
        return src_mask
    
    def get_tgt_mask(self, tgt: torch.Tensor):
        N, len_tgt = tgt.shape
        tgt_mask = torch.tril(torch.ones(len_tgt, len_tgt)).expand(N, 1, len_tgt, len_tgt)  # for each training example
        return tgt_mask.to(self.device)
