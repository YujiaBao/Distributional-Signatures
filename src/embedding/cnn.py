import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding.wordebd import WORDEBD
from embedding.auxiliary.factory import get_embedding
from collections import OrderedDict


class CNN(nn.Module):
    '''
        An aggregation method that encodes every document through different
        convolution filters (followed by max-over-time pooling).
    '''
    def __init__(self, ebd, args):
        super(CNN, self).__init__()
        self.args = args

        self.ebd = ebd
        self.aux = get_embedding(args)

        self.input_dim = self.ebd.embedding_dim + self.aux.embedding_dim

        # Convolution
        self.convs = nn.ModuleList([nn.Conv1d(
                    in_channels=self.input_dim,
                    out_channels=args.cnn_num_filters,
                    kernel_size=K) for K in args.cnn_filter_sizes])

        # used for visualization
        if args.mode == 'visualize':
            self.scores = [[] for _ in args.cnn_filter_sizes]

        self.ebd_dim = args.cnn_num_filters * len(args.cnn_filter_sizes)

    def _conv_max_pool(self, x, conv_filter=None, weights=None):
        '''
        Compute sentence level convolution
        Input:
            x:      batch_size, max_doc_len, embedding_dim
        Output:     batch_size, num_filters_total
        '''
        assert(len(x.size()) == 3)

        x = x.permute(0, 2, 1)  # batch_size, embedding_dim, doc_len
        x = x.contiguous()

        # Apply the 1d conv. Resulting dimension is
        # [batch_size, num_filters, doc_len-filter_size+1] * len(filter_size)
        assert(not ((conv_filter is None) and (weights is None)))
        if conv_filter is not None:
            x = [conv(x) for conv in conv_filter]

        elif weights is not None:
            x = [F.conv1d(x, weight=weights['convs.{}.weight'.format(i)],
                          bias=weights['convs.{}.bias'.format(i)])
                 for i in range(len(self.args.cnn_filter_sizes))]

        # max pool over time. Resulting dimension is
        # [batch_size, num_filters] * len(filter_size)
        x = [F.max_pool1d(sub_x, sub_x.size(2)).squeeze(2) for sub_x in x]

        # concatenate along all filters. Resulting dimension is
        # [batch_size, num_filters_total]
        x = torch.cat(x, 1)
        x = F.relu(x)

        return x

    def forward(self, data, weights=None):
        '''
            @param data dictionary
                @key text: batch_size * max_text_len
            @param weights placeholder used for maml

            @return output: batch_size * embedding_dim
        '''
        # Apply the word embedding, result:  batch_size, doc_len, embedding_dim
        ebd = self.ebd(data, weights)

        # add augmented embedding if applicable
        aux = self.aux(data, weights)

        ebd = torch.cat([ebd, aux], dim=2)

        # apply 1d conv + max pool, result:  batch_size, num_filters_total
        if weights is None:
            ebd = self._conv_max_pool(ebd, conv_filter=self.convs)
        else:
            ebd = self._conv_max_pool(ebd, weights=weights)

        # update max scores
        if self.args.mode == 'visualize':
            for i, s in enumerate(self.compute_score(data)):
                self.scores[i].append(torch.max(s).item())

        return ebd

    def compute_score(self, data, normalize=False):
        # preparing the input
        ebd = self.ebd(data)
        aux = self.aux(data)
        # (batch_size, doc_len, input_dim)
        x = torch.cat([ebd, aux], dim=-1).detach()

        # (out_channels, in_channels, kernel_size)
        w = [c.weight.data for c in self.convs]
        # (kernel_size, out_channels, in_channels)
        w = [c.permute(2,0,1) for c in w]
        # (out_channels * kernel_size, in_channels)
        w = [c.reshape(-1, self.input_dim) for c in w]

        # (batch_size, doc_len, out_channels * kernel_size)
        x = [x @ c.t() for c in w]
        # (batch_size, doc_len)
        x = [F.max_pool1d(z, z.shape[-1]).squeeze(-1) for z in x]

        if normalize:
            x = [x / np.mean(s) for x, s in zip(x, self.scores)]

        return x
