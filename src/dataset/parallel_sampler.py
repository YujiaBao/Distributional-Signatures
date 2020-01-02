import time
import datetime
from multiprocessing import Process, Queue, cpu_count

import torch
import numpy as np
# from pytorch_transformers import BertModel
from transformers import BertModel

import dataset.utils as utils
import dataset.stats as stats


class ParallelSampler():
    def __init__(self, data, args, num_episodes=None):
        self.data = data
        self.args = args
        self.num_episodes = num_episodes

        self.all_classes = np.unique(self.data['label'])
        self.num_classes = len(self.all_classes)
        if self.num_classes < self.args.way:
            raise ValueError("Total number of classes is less than #way.")

        self.idx_list = []
        for y in self.all_classes:
            self.idx_list.append(
                    np.squeeze(np.argwhere(self.data['label'] == y)))

        self.count = 0
        self.done_queue = Queue()

        self.num_cores = cpu_count() if args.n_workers is 0 else args.n_workers

        self.p_list = []
        for i in range(self.num_cores):
            self.p_list.append(
                    Process(target=self.worker, args=(self.done_queue,)))

        for i in range(self.num_cores):
            self.p_list[i].start()

    def get_epoch(self):
        for _ in range(self.num_episodes):
            # wait until self.thread finishes
            support, query = self.done_queue.get()

            # convert to torch.tensor
            support = utils.to_tensor(support, self.args.cuda, ['raw'])
            query = utils.to_tensor(query, self.args.cuda, ['raw'])

            if self.args.meta_w_target:
                if self.args.meta_target_entropy:
                    w = stats.get_w_target(
                            support, self.data['vocab_size'],
                        self.data['avg_ebd'], self.args.meta_w_target_lam)
                else:  # use rr approxmation (this one is faster)
                    w = stats.get_w_target_rr(
                            support, self.data['vocab_size'],
                        self.data['avg_ebd'], self.args.meta_w_target_lam)
                support['w_target'] = w.detach()
                query['w_target'] = w.detach()

            support['is_support'] = True
            query['is_support'] = False

            yield support, query

    def worker(self, done_queue):
        '''
            Generate one task (support and query).
            Store into self.support[self.cur] and self.query[self.cur]
        '''
        while True:
            if done_queue.qsize() > 100:
                time.sleep(1)
                continue
            # sample ways
            sampled_classes = np.random.permutation(
                    self.num_classes)[:self.args.way]

            source_classes = []
            for j in range(self.num_classes):
                if j not in sampled_classes:
                    source_classes.append(self.all_classes[j])
            source_classes = sorted(source_classes)

            # sample examples
            support_idx, query_idx = [], []
            for y in sampled_classes:
                tmp = np.random.permutation(len(self.idx_list[y]))
                support_idx.append(
                        self.idx_list[y][tmp[:self.args.shot]])
                query_idx.append(
                        self.idx_list[y][
                            tmp[self.args.shot:self.args.shot+self.args.query]])

            support_idx = np.concatenate(support_idx)
            query_idx = np.concatenate(query_idx)
            if self.args.mode == 'finetune' and len(query_idx) == 0:
                query_idx = support_idx

            # aggregate examples
            max_support_len = np.max(self.data['text_len'][support_idx])
            max_query_len = np.max(self.data['text_len'][query_idx])

            support = utils.select_subset(self.data, {}, ['text', 'text_len', 'label'],
                                     support_idx, max_support_len)
            query = utils.select_subset(self.data, {}, ['text', 'text_len', 'label'],
                                   query_idx, max_query_len)

            if self.args.embedding in ['idf', 'meta', 'meta_mlp']:
                # compute inverse document frequency over the meta-train set
                idf = stats.get_idf(self.data, source_classes)
                support['idf'] = idf
                query['idf'] = idf

            if self.args.embedding in ['iwf', 'meta', 'meta_mlp']:
                # compute SIF over the meta-train set
                iwf = stats.get_iwf(self.data, source_classes)
                support['iwf'] = iwf
                query['iwf'] = iwf

            if 'pos' in self.args.auxiliary:
               support = utils.select_subset(
                       self.data, support, ['head', 'tail'], support_idx)
               query = utils.select_subset(
                       self.data, query, ['head', 'tail'], query_idx)

            done_queue.put((support, query))

    def __del__(self):
        '''
            Need to terminate the processes when deleting the object
        '''
        for i in range(self.num_cores):
            self.p_list[i].terminate()

        del self.done_queue
