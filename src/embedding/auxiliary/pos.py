import torch
import torch.nn as nn
import torch.nn.functional as F


class POS(nn.Module):
    '''
        Embedding module that combines position-aware embedding
        and standard text embedding.

        Position embedding should only be used with CNN or META
        (sentences are of variable length)
    '''
    def __init__(self, args):
        super(POS, self).__init__()
        self.args = args

        self.embedding_dim = 2 * args.pos_ebd_dim

        # position embedding
        # 2 * length to account for -length to +length
        self.pos1 = nn.Embedding(
                2 * args.pos_max_len, args.pos_ebd_dim, padding_idx=0)
        self.pos2 = nn.Embedding(
                2 * args.pos_max_len, args.pos_ebd_dim, padding_idx=0)


    def forward(self, data, weights=None):
        text = data['text']
        head = data['head'].t()  # (2, n) where [0] is start and [1] is end
        tail = data['tail'].t()  # (2, n)

        assert head.shape[1] == tail.shape[1] == len(text)
        n = head.shape[1]
        max_len = max(data['text_len'])

        # (n, max_len)
        idx = torch.arange(max_len, device=data['text'].device).expand(n, -1)
        # (max_len, 1)
        h0, h1 = head[0].unsqueeze(1), head[1].unsqueeze(1)
        t0, t1 = tail[0].unsqueeze(1), tail[1].unsqueeze(1)
        # filler
        zero = torch.tensor(0, device=data['text'].device)

        # (n, max_len) + add max_len to center 0
        pos1 = torch.where(idx < h0, idx - h0, zero) + \
               torch.where(idx > h1, idx - h1, zero) + self.args.pos_max_len
        pos2 = torch.where(idx < t0, idx - t0, zero) + \
               torch.where(idx > t1, idx - t1, zero) + self.args.pos_max_len

        if weights is None:
            return torch.cat([self.pos1(pos1), self.pos2(pos2)], dim=2)
        else:
            return torch.cat([
                F.embedding(pos1, weights['aux.aux.0.pos1.weight']),
                F.embedding(pos2, weights['aux.aux.0.pos2.weight'])
            ], dim=2)
