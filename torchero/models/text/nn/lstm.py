import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMForTextClassification(nn.Module):
    @classmethod
    def from_config(cls, config):
        return cls(config['vocab_size'],
                   config['output_size'],
                   config['embedding_dim'],
                   config['hidden_size'],
                   config.get('num_layers', 1),
                   config.get('bidirectional', False),
                   config.get('mode', 'max'),
                   config.get('dropout_clf', 0.5))

    @property
    def config(self):
        return {
            'vocab_size': self.embeddings.num_embeddings,
            'output_size': self.linear.out_features,
            'embedding_dim': self.embeddings.embedding_dim,
            'hidden_size': self.lstm.hidden_size,
            'bidirectional': self.lstm.bidirectional,
            'num_layers': self.lstm.num_layers,
            'mode': self.mode,
            'dropout_clf': self.dropout_clf.p,
        }


    def __init__(self,
                 vocab_size,
                 output_size,
                 embedding_dim,
                 hidden_size,
                 num_layers=1,
                 bidirectional=False,
                 mode='max',
                 dropout_clf=0.5,
                 padding_idx=0):
        super(LSTMForTextClassification, self).__init__()
        self.embeddings = nn.Embedding(vocab_size,
                                       embedding_dim=embedding_dim,
                                       padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.mode = mode
        if self.mode == 'max':
            self.reduce = self._pool
        elif self.mode == 'last':
            self.reduce = self._last
        else:
            raise ValueError("Invalid mode. Choose from: 'max', 'last'")
        self.dropout_clf = nn.Dropout(dropout_clf)
        self.linear = nn.Linear(hidden_size*2 if bidirectional else hidden_size,
                                output_size)

    def _pool(self, output_layers, _):
        x, _ = pad_packed_sequence(output_layers, batch_first=True)
        x, _ = torch.max(x, dim=1)
        return x

    def _last(self, _, x):
        return x[-1]

    def forward(self, x, lens):
        lens = lens.cpu()
        x = self.embeddings(x)
        x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        output_layers, (h, c) = self.lstm(x)
        x = self.reduce(output_layers, h)
        x = self.dropout_clf(x)
        x = self.linear(x)
        return x
