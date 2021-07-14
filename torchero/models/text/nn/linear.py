import torch
from torch import nn

class LinearModel(nn.Module):
    @classmethod
    def from_config(cls, config):
        return cls(config['vocab_size'],
                   config['embedding_dim'],
                   config['output_size'])

    @property
    def config(self):
        return {'vocab_size': self.embeddings.num_embeddings,
                'embedding_dim': self.embeddings.embedding_dim,
                'output_size': self.linear.out_features}

    def __init__(self, vocab_size, embedding_dim, output_size):
        super(LinearModel, self).__init__()
        self.embeddings = nn.EmbeddingBag(vocab_size,
                                          embedding_dim=embedding_dim)
        self.linear = nn.Linear(embedding_dim, output_size)

    def forward(self, x, lens):
        x = torch.cat([e[:l] for e, l in zip(x, lens)])
        offsets = torch.cat([torch.zeros(1, dtype=lens.dtype, device=lens.device),
                             lens[:-1]]).cumsum(dim=0)
        x = self.embeddings(x, offsets)
        x = self.linear(x)
        return x
