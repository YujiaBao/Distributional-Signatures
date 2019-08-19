import os
from collections import defaultdict
from tqdm import tqdm
from termcolor import colored
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from math import isnan


def _subset_selection(data, classes):
    '''
        Filter out examples in the data dictionary that do not belong to the
        list of classes.
    '''
    idx = []
    for y in classes:
        idx.append(data['label'] == y)
    idx = np.any(idx, axis=0)

    return {
            'text': data['text'][idx],
            'text_len': data['text_len'][idx],
            'label': data['label'][idx],
            'raw': data['raw'][idx],
            'vocab_size': data['vocab_size']
            }


def _compute_idf(data, classes=None):
    '''
        Compute idf over the train data
        Compute the statistics during the first run
    '''
    data_len = len(data['label'])

    if 'n_d' not in data:
        # finding documents belong to each class label
        unique_text = defaultdict(list)
        for i in range(data_len):
            unique_text[data['label'][i]].append(np.unique(data['text'][i,:]))

        # computing num of occurrences for each class label
        data['n_d'] = {}
        for key, value in unique_text.items():
            total_text = np.concatenate(value)
            idx, counts = np.unique(total_text, return_counts=True)
            data['n_d'][key] = (idx, counts)

    if classes is None:
        classes = np.unique(data['label'])

    n_t = np.zeros(data['vocab_size'], dtype=np.float32)
    for key in classes:
        n_t[data['n_d'][key][0]] += data['n_d'][key][1]

    idf = np.log(data_len / (1.0 + n_t))
    idf[idf<0] = 0
    idf = np.expand_dims(idf, axis=1)  # convert to 2d

    return idf


def _compute_iwf(data, classes=None):
    '''
        Compute sif features over the train data
        Compute the statistics during the first run
    '''
    data_len = len(data['label'])

    if 'n_t' not in data:
        # finding documents belong to each class label
        data['n_t'] = {}
        for i in range(data_len):
            idx, counts = np.unique(data['text'][i,:], return_counts=True)
            if data['label'][i] not in data['n_t']:
                data['n_t'][data['label'][i]] = np.zeros(
                        data['vocab_size'], dtype=np.float32)

            data['n_t'][data['label'][i]][idx] += counts


    if classes is None:
        classes = np.unique(data['label'])

    n_tokens = np.zeros((len(classes), data['vocab_size']), dtype=np.float32)
    for i, key in enumerate(classes):
        n_tokens[i,:] = data['n_t'][key]

    n_tokens_sum = np.sum(n_tokens, axis=0, keepdims=True)
    n_total = np.sum(n_tokens_sum)

    p_t = n_tokens_sum / n_total

    # compute iwf
    iwf = 1e-5 / (1e-5 + p_t)
    iwf = np.transpose(iwf)

    return iwf


def precompute_stats(train_data, val_data, test_data, args):
    '''
    Compute idf and iwf over the training data
    '''
    if args.embedding in ['idf', 'meta', 'meta_mlp']:
        idf = _compute_idf(train_data)

        train_data['idf'] = idf
        val_data['idf'] = idf
        test_data['idf'] = idf

    if args.embedding in ['iwf', 'meta', 'meta_mlp']:
        iwf = _compute_iwf(train_data)
        train_data['iwf'] = iwf
        val_data['iwf'] = iwf
        test_data['iwf'] = iwf


def get_idf(data, source_classes):
    '''
        return idf computed over the source classes.
        if data is not train_data (so it is either val or test), return the idf
        pre-computed over the train_data
    '''
    return _compute_idf(data, source_classes) if 'is_train' in data else data['idf']


def get_iwf(data, source_classes):
    '''
        return itf computed over the source classes.
        if data is not train_data (so it is either val or test), return the itf
        pre-computed over the train_data
    '''
    return _compute_iwf(data, source_classes) if 'is_train' in data else data['iwf']


def get_w_target_rr(data, vocab_size, ebd_model, w_target_lam):
    '''
        Compute the importance of every tokens in the support set
        Convert to Ridge Regression as it admits analytical solution.
        Using this explicit formula improve speed by 2x

        @return w: vocab_size * num_classes
    '''

    text_ebd = ebd_model(data)
    label = data['label'].clone()

    unique, inv_idx = torch.unique(label, sorted=True, return_inverse=True)
    new_label = torch.arange(len(unique), dtype=unique.dtype, device=unique.device)

    label = new_label[inv_idx]
    label_onehot = F.embedding(label,
            torch.eye(len(unique), dtype=torch.float, device=text_ebd.device))

    I = torch.eye(len(text_ebd), dtype=torch.float, device=text_ebd.device)

    w = text_ebd.t()\
            @ torch.inverse(text_ebd @ text_ebd.t() + w_target_lam * I)\
            @ label_onehot

    return w


def get_w_target(data, vocab_size, ebd_model, w_target_lam):
    '''
        Compute the importance of every tokens in the support set
        A simple softmax classifier with L2 penalty

        @return w: vocab_size * num_classes
    '''

    text_ebd = ebd_model(data)
    label = data['label'].clone()

    unique, inv_idx = torch.unique(label, sorted=True, return_inverse=True)
    new_label = torch.arange(len(unique), dtype=unique.dtype, device=unique.device)
    label = new_label[inv_idx]

    ebd_dim = text_ebd.size()[-1]
    num_classes = len(unique)
    device = torch.device(text_ebd.device)


    def init_w_b(ebd_dim, num_classes, device):
        w = torch.rand((ebd_dim, num_classes), dtype=torch.float,
                requires_grad=True, device=device)
        b = torch.rand(num_classes, dtype=torch.float, requires_grad=True,
                device=device)

        return w, b

    w, b = init_w_b(ebd_dim, num_classes, device)

    lr = 1e-1

    opt = torch.optim.Adam([w, b], lr=lr)
    i = 0
    while True:
        opt.zero_grad()

        pred = text_ebd @ w + b.unsqueeze(0)
        wnorm = w.norm()
        loss = F.cross_entropy(pred, label) + w_target_lam * (wnorm** 2)
        loss.backward()

        grad = w.grad.data.norm().item()
        # acc = torch.mean((torch.argmax(pred, dim=1) == label).float()).item()
        # norm = wnorm.item()
        # print('iter {:>4g}, acc {:.2f}, grad {:.4f}, norm {:.2f}'.format(
        #     i, acc, grad, norm))

        if grad < 1e-1:
            break

        opt.step()

        if torch.isnan(torch.sum(w)) or i == 1e5:
            # need to restart with a smaller learning rate
            acc = torch.mean((torch.argmax(pred, dim=1) == label).float()).item()
            norm = wnorm.item()
            print('iter {:>4g}, acc {:.2f}, grad {:.4f}, norm {:.2f}'.format(
                i, acc, grad, norm))

            lr *= 0.1
            w, b = init_w_b(ebd_dim, num_classes, device)
            opt = torch.optim.Adam([w, b], lr=lr)

            i = -1

        i += 1

    return w
