import torch
from classifier.base import BASE


class NN(BASE):
    '''
        Nearest neighbour classifier
    '''
    def __init__(self, ebd_dim, args):
        super(NN, self).__init__(args)
        self.ebd_dim = ebd_dim

    def forward(self, XS, YS, XQ, YQ):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return None (a placeholder for loss)
        '''
        if self.args.nn_distance == 'l2':
            dist = self._compute_l2(XS, XQ)
        elif self.args.nn_distance == 'cos':
            dist = self._compute_cos(XS, XQ)
        else:
            raise ValueError("nn_distance can only be l2 or cos.")

        # 1-NearestNeighbour
        nn_idx = torch.argmin(dist, dim=1)
        pred = YS[nn_idx]

        acc = torch.mean((pred == YQ).float()).item()

        return acc, None
