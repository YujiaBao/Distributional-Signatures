import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

from classifier.base import BASE


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias = False)
        # split the weight update component to direction and norm
        # WeightNorm.apply(self.L, 'weight', dim=0)

        # a fixed scale factor to scale the output of cos value
        # into a reasonably large input for softmax
        self.scale_factor = 10;

    def forward(self, x):

        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        # L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)

        # self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)

        cos_dist = self.L(x_normalized)  # matrix product by forward function
        scores = self.scale_factor * (cos_dist)

        return scores


class MLP(BASE):
    def __init__(self, ebd_dim, args, top_layer=None):
        super(MLP, self).__init__(args)

        self.ebd_dim = ebd_dim

        self.mlp = self._init_mlp(
                ebd_dim, self.args.mlp_hidden, self.args.dropout)

        self.top_layer = top_layer

    @staticmethod
    def get_top_layer(args, n_classes):
        '''
            Creates final layer of desired type
            @return final classification layer
        '''
        loss_type = args.finetune_loss_type

        if loss_type == 'softmax':
            return nn.Linear(args.mlp_hidden[-1], n_classes)
        elif loss_type == 'dist':
            return distLinear(args.mlp_hidden[-1], n_classes)

    def forward(self, XS, YS=None, XQ=None, YQ=None, weights=None):
        '''
            if y is specified, return loss and accuracy
            otherwise, return the transformed x

            @param: XS: batch_size * input_dim
            @param: YS: batch_size (optional)

            @return: XS: batch_size * output_dim
        '''
        # normal training procedure, train stage only use query
        if weights is None:
            XS = self.mlp(XS)
        else:
            # find weight and bias keys for the mlp module
            w_keys, b_keys = [], []
            for key in weights.keys():
                if key[:4] == 'mlp.':
                    if key[-6:] == 'weight':
                        w_keys.append(key)
                    else:
                        b_keys.append(key)

            for i in range(len(w_keys)-1):
                XS = F.dropout(XS, self.args.dropout, training=self.training)
                XS = F.linear(XS, weights[w_keys[i]], weights[b_keys[i]])
                XS = F.relu(XS)

            XS = F.dropout(XS, self.args.dropout, training=self.training)
            XS = F.linear(XS, weights[w_keys[-1]], weights[b_keys[-1]])

        if self.top_layer is not None:
            XS = self.top_layer(XS)


        # normal training procedure, compute loss/acc
        if YS is not None:
            _, YS = torch.unique(YS, sorted=True, return_inverse=True)
            loss = F.cross_entropy(XS, YS)
            acc = BASE.compute_acc(XS, YS)

            return acc, loss

        else:
            return XS
