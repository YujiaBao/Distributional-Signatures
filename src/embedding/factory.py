import torch
import datetime

from embedding.wordebd import WORDEBD
from embedding.cxtebd import CXTEBD

from embedding.avg import AVG
from embedding.cnn import CNN
from embedding.idf import IDF
from embedding.meta import META


def get_embedding(vocab, args):
    print("{}, Building embedding".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')))

    # check if loading pre-trained embeddings
    if args.bert:
        ebd = CXTEBD()
    else:
        ebd = WORDEBD(vocab)

    if args.embedding == 'avg':
        model = AVG(ebd, args)
    elif args.embedding in ['idf', 'iwf']:
        model = IDF(ebd, args)
    elif args.embedding in ['meta', 'meta_mlp']:
        model = META(ebd, args)
    elif args.embedding == 'cnn':
        model = CNN(ebd, args)

    if args.snapshot != '':
        # load pretrained models
        print("{}, Loading pretrained embedding from {}".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
            args.snapshot + '.ebd'
            ))
        model.load_state_dict(torch.load(args.snapshot + '.ebd'))

    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model
