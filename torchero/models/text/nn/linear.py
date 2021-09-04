import torch
from torch import nn
from torchero.utils.text import vectors, transforms

class LinearModel(nn.Module):
    arch_names = [
        'w2v-googlenews-300',
        'glove-en-wiki-50',
        'glove-en-wiki-100',
        'glove-en-wiki-200',
        'glove-en-wiki-300',
        'glove-twitter-25',
        'glove-twitter-50',
        'glove-twitter-100',
        'glove-twitter-200',
        'glove-common-crawl',
        'glove-common-crawl-large'
    ]

    @classmethod
    def from_pretrained(cls,
                        arch,
                        output_size,
                        root=None,
                        tokenizer='basic',
                        max_len=None,
                        truncate_mode='left'):
        """ Create a Linear Model from pretrained word vectors

        Arguments:
            arch (str): Word vectors name.
            output_size (int): Number of outputs
            root (str, PathLike, optional): Where to download the word vectors
            tokenizer (str): Tokenizer to use
            max_len (int): Maximum number of tokens per sample
            truncate_mode (str, ``torch.utils.text.transforms.Truncator):
                Truncation method to use. If a string is passed it should be one of
                'left', 'center', 'right'.
        Returns:
            A tuple (``torchero.utils.text.transforms.Compose``, ``torchero.models.text.nn.LinearModel``)
            with the text prpeprocessing pipeline and the LinearModel
        """
        if arch not in cls.arch_names:
            raise ValueError("{arch} is not a valid architecture. "
                             "Choose from {names}".format(arch=arch,
                                                          names=','.join(map(repr, cls.arch_names))))
        arch = arch.split('-')
        vec_type, vec_domain, vec_dim = arch[0], '-'.join(arch[1:-1]), int(arch[-1])
        if vec_type == 'w2v':
            wv = vectors.Word2VecVectors.from_url(domain=vec_domain, root=root)
        elif vec_type == 'glove':
            wv = vectors.GLoVeVectors.from_url(domain=vec_domain, dim=vec_dim, root=root)
        transform = transforms.basic_text_transform(pre_process=None,
                                                    vocab=wv.vocab,
                                                    tokenizer=tokenizer,
                                                    max_len=max_len,
                                                    truncate_mode=truncate_mode)

        model = cls(vocab_size=len(wv),
                    embedding_dim=wv.vector_size,
                    output_size=output_size)
        wv.replace_embeddings(transform.vocab, model.embeddings, freeze=False)
        return transform, model


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
        return x.squeeze(-1)
