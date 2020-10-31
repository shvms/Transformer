import torch

from Transformer import Transformer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src = torch.tensor(
        [[1, 5, 6, 4, 3, 9, 5, 2, 0],
         [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    tgt = torch.tensor(
        [[1, 7, 4, 3, 5, 9, 2, 8],
         [1, 5, 6, 2, 4, 7, 6, 2]]
    ).to(device)

    src_pad_idx, tgt_pad_idx = 0, 0
    src_vocab_size, tgt_vocab_size = 10, 10
    model = Transformer(src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx, device=device).to(device)
    out = model(src, tgt[:, :-1])
    print(out.shape)
    print(out)

if __name__ == '__main__':
    main()
