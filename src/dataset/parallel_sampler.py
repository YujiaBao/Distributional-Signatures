import time
import datetime
from multiprocessing import Process, Queue, cpu_count

import torch
import numpy as np
from pytorch_transformers import BertModel

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

        # print("{}, Initializing parallel data loader with {} processes".format(
        #     datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
        #     self.num_cores), flush=True)

        if self.args.bert_path is not None:
            # use bert online
            print("{}, Loading pretrained bert from {}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                self.args.bert_path), flush=True)

            if self.args.cuda != -1:
                self.model = BertModel.from_pretrained(
                        'bert-base-uncased',
                        cache_dir=self.args.bert_path).cuda(self.args.cuda)
            else:
                self.model = BertModel.from_pretrained(
                        'bert-base-uncased',
                        cache_dir=self.args.bert_path)

            self.model.eval()

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

            if 'bert_id' in support.keys():
                # run bert to get ebd
                support['ebd'] = self.get_bert(
                        support['bert_id'],
                        support['text_len']+2)
                query['ebd'] = self.get_bert(
                        query['bert_id'],
                        query['text_len']+2)

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

            if self.args.bert:
                # prepare bert token id
                # +2 becuase bert_id includes [CLS] and [SEP]
                support = utils.select_subset(self.data, support, ['bert_id'],
                    support_idx, max_support_len+2)
                query = utils.select_subset(self.data, query, ['bert_id'],
                    query_idx, max_query_len+2)


            if 'pos' in self.args.auxiliary:
               support = utils.select_subset(
                       self.data, support, ['head', 'tail'], support_idx)
               query = utils.select_subset(
                       self.data, query, ['head', 'tail'], query_idx)

            done_queue.put((support, query))

    def get_bert(self, bert_id, text_len, mini_batch_size=25):
        '''
            Return the last layer of bert's representation
            @param: bert_id: batch_size * max_text_len+2
            @param: text_len: text_len

            @return: last_layer: batch_size * max_text_len
        '''
        len_range = torch.arange(bert_id.size()[-1], device=bert_id.device,
                dtype=text_len.dtype).expand(*bert_id.size())

        # mask for the bert
        mask1 = (len_range < text_len.unsqueeze(-1)).long()
        # mask for the sep
        mask2 = (len_range < (text_len-1).unsqueeze(-1)).float().unsqueeze(-1)

        # need to use smaller batches
        result = []
        num_batches = int((len(bert_id)-1)/mini_batch_size) + 1

        for i in range(num_batches):
            start_idx = i*mini_batch_size
            end_idx = min((i+1)*mini_batch_size, len(bert_id))
            out = self.model(
                    bert_id[start_idx:end_idx],
                    attention_mask=mask1[start_idx:end_idx])

            last_layer = mask2[start_idx:end_idx] * out[0]

            result.append(last_layer[:,1:-1,:].detach())

        return torch.cat(result, dim=0)

    def __del__(self):
        '''
            Need to terminate the processes when deleting the object
        '''
        for i in range(self.num_cores):
            self.p_list[i].terminate()

        del self.done_queue
