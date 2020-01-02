import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.base import BASE


class ROUTING(BASE):
    '''
        Induction and Relation module of
        "Induction Networks for Few-Shot Text Classification"
    '''
    def __init__(self, ebd_dim, args):
        super(ROUTING, self).__init__(args)
        self.args = args

        self.ebd_dim = ebd_dim

        h = args.induct_hidden_dim
        self.iter = args.induct_iter

        if self.args.embedding == 'meta':
            print('No relation module. Use Prototypical network style prediction')
        else:  # follow the original paper
            self.Ws = nn.Linear(self.ebd_dim, self.ebd_dim)
            self.M = nn.Parameter(torch.Tensor(h, 1, 1, self.ebd_dim, self.ebd_dim).uniform_(-0.1,0.1))
            self.rel = nn.Linear(h, 1)

    def _squash(self, X):
        '''
            Perform squashing over the last dimension
            The dimension remain the same
        '''
        X_norm = torch.norm(X, dim=-1, keepdim=True)

        out = (X_norm ** 2) / (1.0 + X_norm ** 2) / X_norm * X

        return out

    def _compute_prototype(self, XS, YS):
        '''
            Compute the prototype for each class by dynamic routing

            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size

            @return prototype: way x ebd_dim
        '''
        # sort YS to make sure classes of the same labels are clustered together
        YS, indices = torch.sort(YS)
        XS = XS[indices]

        # squash
        if self.args.embedding == 'meta':
            # do not transform the matrix to preserve information when
            # distributional signatures are used
            XS_hat = self._squash(XS)
        else:
            # original paper's transformation
            XS_hat = self._squash(self.Ws(XS))

        b = torch.zeros([self.args.way, self.args.shot], device=XS.device)
        prototype = []
        for it in range(self.iter):
            # perform dynamic routing for each class
            d = F.softmax(b, dim=-1)
            new_b = torch.zeros_like(b)

            for i in range(self.args.way):
                # examples belonging to class i
                XS_hat_cur = XS_hat[i*self.args.shot:(i+1)*self.args.shot,:]

                # generate prototypes
                c_hat = torch.sum(d[i, :].unsqueeze(1) * XS_hat_cur, dim=0)
                c = self._squash(c_hat)

                # update b
                new_b[i,:] = b[i,:] + (XS_hat_cur @ c.unsqueeze(1)).squeeze(1)

                if it == self.iter-1:
                    prototype.append(c.unsqueeze(0))

            b = new_b

        prototype = torch.cat(prototype, dim=0)

        return prototype

    def _compute_relation_score(self, prototype, XQ):
        '''
            Compute the relation score between each prototype and each query
            example

            @param prototype: way x ebd_dim
            @param XQ: query_size x ebd_dim

            @return score: query_size x way
        '''
        prototype = prototype.unsqueeze(0).unsqueeze(0).unsqueeze(-2)
        # 1, 1, way, 1, ebd_dim
        XQ = XQ.unsqueeze(1).unsqueeze(-1).unsqueeze(0)
        # 1, query_size, 1, ebd_dim, 1

        score = torch.matmul(torch.matmul(prototype, self.M),
                             XQ)
        # h, query_size, way, 1, 1

        score = score.squeeze(-1).squeeze(-1).permute(1, 2, 0)
        # query_size, way, h

        score = F.relu(score)
        score = torch.sigmoid(self.rel(score)).squeeze(-1)

        return score

    def forward(self, XS, YS, XQ, YQ):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return loss
        '''

        YS, YQ = self.reidx_y(YS, YQ)

        prototype = self._compute_prototype(XS, YS)

        if self.args.embedding == 'meta':
            # use parameter free comparison when distributional signatures are
            # used
            score = -self._compute_l2(prototype, XQ)
            # score = -self._compute_cos(prototype, XQ)
            # l2 and cos deosn't have much diff empirically across the 6
            # datasets

            loss = F.cross_entropy(score, YQ)

        else:
            # implementation based on the original paper
            score = self._compute_relation_score(prototype, XQ)

            # use regression as training objective
            YQ_onehot = self._label2onehot(YQ)

            loss = torch.sum((YQ_onehot.float() - score) ** 2)

        acc = BASE.compute_acc(score, YQ)

        return acc, loss
