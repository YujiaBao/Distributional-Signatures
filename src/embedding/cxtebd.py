import datetime

import torch
import torch.nn as nn
from transformers import BertModel


class CXTEBD(nn.Module):
    '''
        An embedding layer directly returns precomputed BERT
        embeddings.
    '''
    def __init__(self, pretrained_model_name_or_path=None, cache_dir=None,
                 finetune_ebd=False, return_seq=False):
        '''
            pretrained_model_name_or_path, cache_dir: check huggingface's codebase for details
            finetune_ebd: finetuning bert representation or not during
            meta-training
            return_seq: return a sequence of bert representations, or [cls]
        '''
        super(CXTEBD, self).__init__()

        self.finetune_ebd = finetune_ebd

        self.return_seq = return_seq

        print("{}, Loading pretrained bert".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

        self.model = BertModel.from_pretrained(pretrained_model_name_or_path,
                                               cache_dir=cache_dir)

        self.embedding_dim = self.model.config.hidden_size
        self.ebd_dim = self.model.config.hidden_size

    def get_bert(self, bert_id, text_len):
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
        out = self.model(bert_id, attention_mask=mask1)

        last_layer = mask2 * out[0]

        if self.return_seq:
            # return seq of bert ebd, dim: batch, text_len, ebd_dim
            # return last_layer[:,1:-1,:]
            return last_layer
        else:
            # return [cls], dim: batch, ebd_dim
            return last_layer[:,0,:]

    def forward(self, data, weights=None):
        '''
            @param data: key 'ebd' = batch_size * max_text_len * embedding_dim
            @return output: batch_size * max_text_len * embedding_dim
        '''
        if self.finetune_ebd:
            return self.get_bert(data['text'], data['text_len'])
        else:
            with torch.no_grad():
                return self.get_bert(data['text'], data['text_len'])
