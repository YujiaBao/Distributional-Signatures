import torch
from classifier.nn import NN
from classifier.proto import PROTO
from classifier.r2d2 import R2D2
from classifier.lrd2 import LRD2
from classifier.mlp import MLP
from classifier.routing import ROUTING
from dataset.utils import tprint


def get_classifier(ebd_dim, args):
    tprint("Building classifier")

    if args.classifier == 'nn':
        model = NN(ebd_dim, args)
    elif args.classifier == 'proto':
        model = PROTO(ebd_dim, args)
    elif args.classifier == 'r2d2':
        model = R2D2(ebd_dim, args)
    elif args.classifier == 'lrd2':
        model = LRD2(ebd_dim, args)
    elif args.classifier == 'routing':
        model = ROUTING(ebd_dim, args)
    elif args.classifier == 'mlp':
        # detach top layer from rest of MLP
        if args.mode == 'finetune':
            top_layer = MLP.get_top_layer(args, args.n_train_class)
            model = MLP(ebd_dim, args, top_layer=top_layer)
        # if not finetune, train MLP as a whole
        else:
            model = MLP(ebd_dim, args)
    else:
        raise ValueError('Invalid classifier. '
                         'classifier can only be: nn, proto, r2d2, mlp.')

    if args.snapshot != '':
        # load pretrained models
        tprint("Loading pretrained classifier from {}".format(
            args.snapshot + '.clf'
            ))
        model.load_state_dict(torch.load(args.snapshot + '.clf'))

    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model
