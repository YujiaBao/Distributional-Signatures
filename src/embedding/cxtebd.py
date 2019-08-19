import torch.nn as nn


class CXTEBD(nn.Module):
    '''
        An embedding layer directly returns precomputed BERT
        embeddings.
    '''
    def __init__(self):
        super(CXTEBD, self).__init__()

        self.embedding_dim = 768  # use bert base uncased by default

    def forward(self, data):
        '''
            @param data: key 'ebd' = batch_size * max_text_len * embedding_dim
            @return output: batch_size * max_text_len * embedding_dim
        '''
        return data['ebd']
