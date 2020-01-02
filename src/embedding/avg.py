import torch
import torch.nn as nn
from embedding.wordebd import WORDEBD

class AVG(nn.Module):
    '''
        An aggregation method that encodes every document by its average word
        embeddings.
    '''
    def __init__(self, ebd, args):
        super(AVG, self).__init__()

        self.ebd = ebd

        self.ebd_dim = self.ebd.embedding_dim


    def forward(self, data, weights=None):
        '''
            @param data dictionary
                @key text: batch_size * max_text_len
            @param weights placeholder used for maml
            @return output: batch_size * embedding_dim
        '''
        ebd = self.ebd(data, weights)

        # count length excluding <pad> and <unk>.
        is_zero = (torch.sum(torch.abs(ebd), dim=2) > 1e-8).float()
        soft_len = torch.sum(is_zero, dim=1, keepdim=True)

        soft_len[soft_len < 1] = 1

        # # don't need to mask out the <pad> tokens, as the embeddings are zero
        ebd = torch.sum(ebd, dim=1)

        ebd = ebd / soft_len

        return ebd
