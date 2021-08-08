import math

import torch
from torch import nn

class PositionalEmbedding(nn.Module):
    """ Positional Encoding for Text classification task from paper
        “Attention Is All You Need“
    """
    def __init__(self, embedding_dim, max_len: int = 5000):
        super(PositionalEmbedding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    @property
    def max_len(self):
        return self.pe.shape[0]

    @property
    def embedding_dim(self):
        return self.pe.shape[1]

    def forward(self, x):
        """
        Args:
            x: positions tensor
        """
        x_shape = x.shape
        x = torch.index_select(self.pe, 0, x.flatten())
        return x.view(x_shape + (self.pe.shape[1],))

class TransformerForTextClassification(nn.Module):
    """ Transformer Encoder for Text classification task from paper
        “Attention Is All You Need“
    """
    @classmethod
    def from_config(cls, config):
        return cls(config['vocab_size'],
                   config['num_outputs'],
                   config['max_position_len'],
                   config['dim_model'],
                   config['num_layers'],
                   config['num_heads'],
                   config['dim_feedforward'],
                   config['emb_dropout'],
                   config['tfm_dropout'])

    @property
    def config(self):
        encoder_layer = self.encoder.layers[0]
        return {
            'vocab_size': self.embedding.num_embeddings,
            'num_outputs': self.clf.out_features,
            'max_position_len': self.pos_embedding.max_len,
            'dim_model': encoder_layer.self_attn.embed_dim,
            'num_layers': len(self.encoder.layers),
            'num_heads': encoder_layer.self_attn.num_heads,
            'dim_feedforward': encoder_layer.linear1.out_features,
            'emb_dropout': self.embedding_dropout.p,
            'tfm_dropout': encoder_layer.dropout.p,
        }

    def __init__(self,
                 vocab_size,
                 num_outputs,
                 max_position_len=5000,
                 dim_model=512,
                 num_layers=6,
                 num_heads=8,
                 dim_feedforward=2048,
                 emb_dropout=0.1,
                 tfm_dropout=0.1):
        """ Constructor

        Arguments:
            vocab_size (int): Size of embedding vocabulary
            num_outputs (int): Number of classes/labels
            max_position_len (int): The maximum sequence length that the model can use (required for positional embeddings).
            dim_model (int): Embedding size
            num_layers (int): Number of sub-sencoder layers to stack
            num_heads (int): Number of attention heads in the multi-headed-attention module. Must be divisible by dim_model
            dim_feedforward (int): Dimention of the feedforward network model of each layer
            emb_dropout (float): Dropout for the embedding module
            tfm_dropout (float): Dropout for the transformer encoder module
        """
        super(TransformerForTextClassification, self).__init__()
        self.pos_embedding = PositionalEmbedding(embedding_dim=dim_model, max_len=max_position_len)
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=dim_model, padding_idx=0)
        self.embedding_dropout = nn.Dropout(p=emb_dropout, inplace=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=tfm_dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.clf = nn.Linear(in_features=dim_model, out_features=num_outputs)
        self.init_weights()

    def forward(self, x, lens):
        max_len = max(lens)
        # x (long) := B X L
        token_embedding = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        pos_embedding = self.pos_embedding(torch.arange(0, max_len, device=token_embedding.device))
        x = self.embedding_dropout(token_embedding + pos_embedding)
        padding_mask = torch.stack([torch.cat([torch.zeros(l), torch.ones(max_len-l)]) for l in lens])>0
        padding_mask = padding_mask.to(x.device)
        # x (float) := B x L x D, padding_mask (bool) := B x L
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        # x (float) := B x L x D
        x = x[:,0,:]
        # x (float) := B x D
        x = self.clf(x)
        # X (float) := B x O
        return x

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
