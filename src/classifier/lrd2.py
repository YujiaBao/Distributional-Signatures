import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.base import BASE


class LRD2(BASE):
    '''
        META-LEARNING WITH DIFFERENTIABLE CLOSED-FORM SOLVERS
    '''
    def __init__(self, ebd_dim, args):
        super(LRD2, self).__init__(args)
        self.ebd_dim = ebd_dim

        self.iters = args.lrd2_num_iters

        # meta parameters to learn
        self.lam = nn.Parameter(torch.tensor(-1, dtype=torch.float))

    def _compute_w(self, XS, YS_inner):
        '''
            Use Newton's method to obtain w from support set XS, YS_inner
            https://github.com/bertinetto/r2d2/blob/master/fewshots/models/lrd2.py
        '''

        for i in range(self.iters):
            # use eta to store w_{i-1}^T X
            if i == 0:
                eta = torch.zeros_like(XS[:,0])  # support_size
            else:
                eta = (XS @ w).squeeze(1)

            mu = torch.sigmoid(eta)
            s = mu * (1 - mu)
            z = eta + (YS_inner - mu) / s
            Sinv = torch.diag(1.0/s)

            # Woodbury with regularization
            w = XS.t() @ torch.inverse(XS @ XS.t() + (10. ** self.lam) * Sinv) @ z.unsqueeze(1)

        return w

    def forward(self, XS, YS, XQ, YQ):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return loss
        '''

        # train with Newton's method on support set
        YS, YQ = self.reidx_y(YS, YQ)
        YS_onehot = self._label2onehot(YS)
        YQ_onehot = self._label2onehot(YQ)

        # 1 vs all
        pred = torch.zeros_like(YQ_onehot)

        for y in range(self.args.way):
            # treat y as positive, all others as negative
            YS_inner = YS_onehot[:, y]

            w = self._compute_w(XS, YS_inner)  # ebd_dim, 1

            pred_inner = XQ @ w  # query_size, 1

            pred[:, y] = pred_inner.squeeze(1)

        loss = F.cross_entropy(pred, YQ)

        acc = BASE.compute_acc(pred, YQ)

        return acc, loss
