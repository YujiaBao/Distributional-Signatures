import torch
import torch.nn.functional as F
import torch.nn as nn
from embedding.wordebd import WORDEBD


class IDF(nn.Module):
    '''
        An aggregation method that encodes every document by its the weighted
        word embeddings.
        The weight is computed by the inverse document frequency over the source
        pool
    '''
    def __init__(self, ebd, args):
        super(IDF, self).__init__()

        self.args = args
        self.ebd = ebd

        self.ebd_dim = self.ebd.embedding_dim


    def forward(self, data, weights=None):
        '''
            @param data dictionary
                @key text: batch_size * max_text_len
                @key text_len: batch_size
                @key idf: vocab_size
            @param weights placeholder used for maml
            @return output: batch_size * embedding_dim
        '''
        ebd = self.ebd(data, weights)

        if self.args.embedding == 'idf':
            score = F.embedding(data['text'], data['idf'])
        elif self.args.embedding == 'iwf':
            score = F.embedding(data['text'], data['iwf'])

        ebd = torch.sum(ebd * score, dim=1)
        ebd = ebd / data['text_len'].unsqueeze(-1).float()

        return ebd
