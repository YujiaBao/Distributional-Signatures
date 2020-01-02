import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from embedding.meta import RNN
from embedding.auxiliary.factory import get_embedding


class LSTMAtt(nn.Module):

    def __init__(self, ebd, args):
        super(LSTMAtt, self).__init__()
        self.args = args

        self.ebd = ebd
        self.aux = get_embedding(args)

        self.input_dim = self.ebd.embedding_dim + self.aux.embedding_dim

        # Default settings in induction encoder
        u = args.induct_rnn_dim
        da = args.induct_att_dim

        self.rnn = RNN(self.input_dim, u, 1, True, 0)

        # Attention
        self.head = nn.Parameter(torch.Tensor(da, 1).uniform_(-0.1, 0.1))
        self.proj = nn.Linear(u*2, da)

        self.ebd_dim = u * 2

    def _attention(self, x, text_len):
        '''
            text:     batch, max_text_len, input_dim
            text_len: batch, max_text_len
        '''
        batch_size, max_text_len, _ = x.size()

        proj_x = torch.tanh(self.proj(x.view(batch_size * max_text_len, -1)))
        att = torch.mm(proj_x, self.head)
        att = att.view(batch_size, max_text_len, 1)  # unnormalized

        # create mask
        idxes = torch.arange(max_text_len, out=torch.cuda.LongTensor(max_text_len,
                                 device=text_len.device)).unsqueeze(0)
        mask = (idxes < text_len.unsqueeze(1)).bool()
        att[~mask] = float('-inf')

        # apply softmax
        att = F.softmax(att, dim=1).squeeze(2)  # batch, max_text_len

        return att

    def forward(self, data):
        """
            @param data dictionary
                @key text: batch_size * max_text_len
            @param weights placeholder used for maml

            @return output: batch_size * embedding_dim
        """

        # Apply the word embedding, result:  batch_size, doc_len, embedding_dim
        ebd = self.ebd(data)

        # add augmented embedding if applicable
        aux = self.aux(data)

        ebd = torch.cat([ebd, aux], dim=2)

        # result: batch_size, max_text_len, embedding_dim

        # apply rnn
        ebd = self.rnn(ebd, data['text_len'])
        # result: batch_size, max_text_len, 2*rnn_dim

        # apply attention
        alpha = self._attention(ebd, data['text_len'])

        # aggregate
        ebd = torch.sum(ebd * alpha.unsqueeze(-1), dim=1)

        return ebd
