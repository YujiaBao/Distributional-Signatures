import os
import time
import datetime
from collections import OrderedDict
import itertools
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from termcolor import colored

from dataset.parallel_sampler import ParallelSampler
from train.utils import named_grad_param, grad_param, get_norm


def _copy_weights(source, target):
    '''
        Copy weights from the source net to the target net
        Only copy weights with requires_grad=True
    '''
    target_dict = target.state_dict()
    for name, p in source.named_parameters():
        if p.requires_grad:
            target_dict[name].copy_(p.data.clone())


def _meta_update(model, total_grad, opt, task, maml_batchsize, clip_grad):
    '''
        Aggregate the gradients in total_grad
        Update the initialization in model
    '''

    model['ebd'].train()
    model['clf'].train()
    support, query = task
    XS = model['ebd'](support)
    pred = model['clf'](XS)
    loss = torch.sum(pred)  # this doesn't matter

    # aggregate the gradients (skip nan)
    avg_grad = {
            'ebd': {key: sum(g[key] for g in total_grad['ebd'] if
                        not torch.sum(torch.isnan(g[key])) > 0)\
                    for key in total_grad['ebd'][0].keys()},
            'clf': {key: sum(g[key] for g in total_grad['clf'] if
                        not torch.sum(torch.isnan(g[key])) > 0)\
                    for key in total_grad['clf'][0].keys()}
            }

    # register a hook on each parameter in the model that replaces
    # the current dummy grad with the meta gradiets
    hooks = []
    for model_name in avg_grad.keys():
        for key, value in model[model_name].named_parameters():
            if not value.requires_grad:
                continue

            def get_closure():
                k = key
                n = model_name
                def replace_grad(grad):
                    return avg_grad[n][k] / maml_batchsize
                return replace_grad

            hooks.append(value.register_hook(get_closure()))

    opt.zero_grad()
    loss.backward()

    ebd_grad = get_norm(model['ebd'])
    clf_grad = get_norm(model['clf'])
    if clip_grad is not None:
        nn.utils.clip_grad_value_(
                grad_param(model, ['ebd', 'clf']), clip_grad)

    opt.step()

    for h in hooks:
        # remove the hooks before the next training phase
        h.remove()

    total_grad['ebd'] = []
    total_grad['clf'] = []

    return ebd_grad, clf_grad


def train(train_data, val_data, model, args):
    '''
        Train the model (obviously~)
    '''
    # creating a tmp directory to save the models
    out_dir = os.path.abspath(os.path.join(
                                  os.path.curdir,
                                  "tmp-runs",
                                  str(int(time.time() * 1e7))))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    best_acc = 0
    sub_cycle = 0
    best_path = None

    opt = torch.optim.Adam(grad_param(model, ['ebd', 'clf']), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, 'max', patience=5, factor=0.1, verbose=True)

    # clone the original model
    fast_model = {
            'ebd': copy.deepcopy(model['ebd']),
            'clf': copy.deepcopy(model['clf']),
            }

    print("{}, Start training".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')))

    train_gen = ParallelSampler(
            train_data, args, args.train_episodes * args.maml_batchsize)
    val_gen = ParallelSampler(val_data, args, args.val_episodes)
    for ep in range(args.train_epochs):
        sampled_tasks = train_gen.get_epoch()

        meta_grad_dict = {'clf': [], 'ebd': []}

        train_episodes = range(args.train_episodes)
        if not args.notqdm:
            train_episodes = tqdm(train_episodes, ncols=80, leave=False,
                                  desc=colored('Training on train', 'yellow'))

        for _ in train_episodes:
            # update the initialization based on a batch of tasks
            total_grad = {'ebd': [], 'clf': []}

            for _ in range(args.maml_batchsize):
                # print('start', flush=True)
                task = next(sampled_tasks)

                # clone the current initialization
                _copy_weights(model['ebd'], fast_model['ebd'])
                _copy_weights(model['clf'], fast_model['clf'])

                # get the meta gradient
                if args.maml_firstorder:
                    train_one_fomaml(task, fast_model, args, total_grad)
                else:
                    train_one(task, fast_model, args, total_grad)

            ebd_grad, clf_grad = _meta_update(
                    model, total_grad, opt, task, args.maml_batchsize,
                    args.clip_grad)
            meta_grad_dict['ebd'].append(ebd_grad)
            meta_grad_dict['clf'].append(clf_grad)

        # evaluate training accuracy
        if ep % 10 == 0:
            acc, std = test(train_data, model, args, args.val_episodes, False,
                            train_gen.get_epoch())
            print("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f} ".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                "ep", ep,
                colored("train", "red"),
                colored("acc:", "blue"), acc, std,
                ), flush=True)

        # evaluate validation accuracy
        cur_acc, cur_std = test(val_data, model, args, args.val_episodes, False,
                                val_gen.get_epoch())
        print(("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f} "
               "{:s} {:s}{:>7.4f}, {:s}{:>7.4f}").format(
               datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
               "ep", ep,
               colored("val  ", "cyan"),
               colored("acc:", "blue"), cur_acc, cur_std,
               colored("train stats", "cyan"),
               colored("ebd_grad:", "blue"),
               np.mean(np.array(meta_grad_dict['ebd'])),
               colored("clf_grad:", "blue"),
               np.mean(np.array(meta_grad_dict['clf']))
               ), flush=True)

        # Update the current best model if val acc is better
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_path = os.path.join(out_dir, str(ep))

            # save current model
            print("{}, Save cur best model to {}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                best_path))

            torch.save(model['ebd'].state_dict(), best_path + '.ebd')
            torch.save(model['clf'].state_dict(), best_path + '.clf')

            sub_cycle = 0
        else:
            sub_cycle += 1

        # Break if the val acc hasn't improved in the past patience epochs
        if sub_cycle == args.patience:
            break

    print("{}, End of training. Restore the best weights".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')))

    # restore the best saved model
    model['ebd'].load_state_dict(torch.load(best_path + '.ebd'))
    model['clf'].load_state_dict(torch.load(best_path + '.clf'))

    if args.save:
        # save the current model
        out_dir = os.path.abspath(os.path.join(
                                      os.path.curdir,
                                      "saved-runs",
                                      str(int(time.time() * 1e7))))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        best_path = os.path.join(out_dir, 'best')

        print("{}, Save best model to {}".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
            best_path), flush=True)

        torch.save(model['ebd'].state_dict(), best_path + '.ebd')
        torch.save(model['clf'].state_dict(), best_path + '.clf')

        with open(best_path + '_args.txt', 'w') as f:
            for attr, value in sorted(args.__dict__.items()):
                f.write("{}={}\n".format(attr, value))

    return


def train_one(task, fast, args, total_grad):
    '''
        Update the fast_model based on the support set.
        Return the gradient w.r.t. initializations over the query set
    '''
    support, query = task

    # map class label into 0,...,num_classes-1
    YS, YQ = fast['clf'].reidx_y(support['label'], query['label'])

    fast['ebd'].train()
    fast['clf'].train()

    # get weights
    fast_weights = {
        'ebd': OrderedDict(
            (name, param) for (name, param) in named_grad_param(fast, ['ebd'])),
        'clf': OrderedDict(
            (name, param) for (name, param) in named_grad_param(fast, ['clf'])),
        }

    num_ebd_w = len(fast_weights['ebd'])
    num_clf_w = len(fast_weights['clf'])

    # fast adaptation
    for i in range(args.maml_innersteps):
        if i == 0:
            XS = fast['ebd'](support)
            pred = fast['clf'](XS)
            loss = F.cross_entropy(pred, YS)
            grads = torch.autograd.grad(loss, grad_param(fast, ['ebd', 'clf']),
                    create_graph=True)

        else:
            XS = fast['ebd'](support, fast_weights['ebd'])
            pred = fast['clf'](XS, weights=fast_weights['clf'])
            loss = F.cross_entropy(pred, YS)
            grads = torch.autograd.grad(
                    loss,
                    itertools.chain(fast_weights['ebd'].values(),
                                   fast_weights['clf'].values()),
                    create_graph=True)

        # update fast weight
        fast_weights['ebd'] = OrderedDict(
                (name, param-args.maml_stepsize*grad) for ((name, param), grad)
                in zip(fast_weights['ebd'].items(), grads[:num_ebd_w]))

        fast_weights['clf'] = OrderedDict(
                (name, param-args.maml_stepsize*grad) for ((name, param), grad)
                in zip(fast_weights['clf'].items(), grads[num_ebd_w:]))

    # forward on the query, to get meta loss
    XQ = fast['ebd'](query, fast_weights['ebd'])
    pred = fast['clf'](XQ, weights=fast_weights['clf'])
    loss = F.cross_entropy(pred, YQ)

    grads = torch.autograd.grad(loss, grad_param(fast, ['ebd', 'clf']))

    grads_ebd = {name: g for ((name, _), g) in zip(
        named_grad_param(fast, ['ebd']),
        grads[:num_ebd_w])}
    grads_clf = {name: g for ((name, _), g) in zip(
        named_grad_param(fast, ['clf']),
        grads[num_ebd_w:])}

    total_grad['ebd'].append(grads_ebd)
    total_grad['clf'].append(grads_clf)

    return


def train_one_fomaml(task, fast, args, total_grad):
    '''
        Update the fast_model based on the support set.
        Return the gradient w.r.t. initializations over the query set
        First order MAML
    '''
    support, query = task

    # map class label into 0,...,num_classes-1
    YS, YQ = fast['clf'].reidx_y(support['label'], query['label'])

    opt = torch.optim.SGD(grad_param(fast, ['ebd', 'clf']),
                          lr=args.maml_stepsize)

    fast['ebd'].train()
    fast['clf'].train()

    # fast adaptation
    for i in range(args.maml_innersteps):
        opt.zero_grad()

        XS = fast['ebd'](support)
        acc, loss = fast['clf'](XS, YS)

        loss.backward()

        opt.step()

    # forward on the query, to get meta loss
    XQ = fast['ebd'](query)
    acc, loss = fast['clf'](XQ, YQ)

    loss.backward()

    grads_ebd = {name: p.grad for (name, p) in named_grad_param(fast, ['ebd'])\
                 if p.grad is not None}  # pooler does not have grad in Bert
    grads_clf = {name: p.grad for (name, p) in named_grad_param(fast, ['clf'])}

    total_grad['ebd'].append(grads_ebd)
    total_grad['clf'].append(grads_clf)

    return


def test(test_data, model, args, num_episodes, verbose=True, sampled_tasks=None):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    # clone the original model
    fast_model = {
            'ebd': copy.deepcopy(model['ebd']),
            'clf': copy.deepcopy(model['clf']),
            }

    if sampled_tasks is None:
        sampled_tasks = ParallelSampler(
                test_data, args, num_episodes).get_epoch()

    acc = []

    sampled_tasks = enumerate(sampled_tasks)
    if not args.notqdm:
        sampled_tasks = tqdm(sampled_tasks, total=num_episodes, ncols=80,
                leave=False, desc=colored('Testing on val', 'yellow'))

    for i, task in sampled_tasks:
        if i == num_episodes and not args.notqdm:
            sampled_tasks.close()
            break
        _copy_weights(model['ebd'], fast_model['ebd'])
        _copy_weights(model['clf'], fast_model['clf'])
        acc.append(test_one(task, fast_model, args))

    acc = np.array(acc)

    if verbose:
        print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                colored("acc mean", "blue"),
                np.mean(acc),
                colored("std", "blue"),
                np.std(acc),
                ))

    return np.mean(acc), np.std(acc)


def test_one(task, fast, args):
    '''
        Evaluate the model on one sampled task. Return the accuracy.
    '''
    support, query = task
    YS, YQ = fast['clf'].reidx_y(support['label'], query['label'])

    fast['ebd'].train()
    fast['clf'].train()

    opt = torch.optim.SGD(grad_param(fast, ['ebd', 'clf']),
                          lr=args.maml_stepsize)

    for i in range(args.maml_innersteps*2):
        XS = fast['ebd'](support)
        pred = fast['clf'](XS)
        loss = F.cross_entropy(pred, YS)

        opt.zero_grad()
        loss.backward()
        opt.step()

    fast['ebd'].eval()
    fast['clf'].eval()

    XQ = fast['ebd'](query)
    pred = fast['clf'](XQ)
    acc = torch.mean((torch.argmax(pred, dim=1) == YQ).float()).item()

    return acc
