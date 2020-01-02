import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding.auxiliary.pos import POS


def get_embedding(args):
    '''
        @return AUX module with aggregated embeddings or None if args.aux
        did not provide additional embeddings
    '''
    print("{}, Building augmented embedding".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')))

    aux = []
    for ebd in args.auxiliary:
        if ebd == 'pos':
            aux.append(POS(args))
        else:
            raise ValueError('Invalid argument for auxiliary ebd')

    if args.cuda != -1:
        aux = [a.cuda(args.cuda) for a in aux]

    model = AUX(aux, args)

    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model


class AUX(nn.Module):
    '''
        Wrapper around combination of auxiliary embeddings
    '''

    def __init__(self, aux, args):
        super(AUX, self).__init__()
        self.args = args
        # this is a list of nn.Module
        self.aux = nn.ModuleList(aux)
        # this is 0 if self.aux is empty
        self.embedding_dim = sum(a.embedding_dim for a in self.aux)

    def forward(self, data, weights=None):
        # torch.cat will discard the empty tensor
        if len(self.aux) == 0:
            if self.args.cuda != -1:
                return torch.FloatTensor().cuda(self.args.cuda)
            return torch.FloatTensor()

        # aggregate results from each auxiliary module
        results = [aux(data, weights) for aux in self.aux]

        # aux embeddings should only be used with cnn, meta or meta_mlp.
        # concatenate together with word embeddings
        assert (self.args.embedding in ['cnn', 'meta', 'meta_mlp', 'lstmatt'])
        x = torch.cat(results, dim=2)

        return x
