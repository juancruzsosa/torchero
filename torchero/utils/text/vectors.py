from io import StringIO
import os
import tempfile
from zipfile import ZipFile
import gzip
import pickle
import struct

import numpy as np
import torch

from torchero.utils.io import download_from_url
from torchero.utils.text.transforms.vocab import Vocab

class KeyedVectors(object):
    """ Class to represent list of word vectors. Each word could be any
    hashable object
    """
    ROOT_DIR = os.path.join(tempfile.gettempdir(), 'torchero', 'vectors')

    @classmethod
    def make_vectors_dir(cls):
        os.makedirs(cls.ROOT_DIR, exist_ok=True)

    @classmethod
    def from_dict(cls, kv):
        return cls(kv.keys(), kv.values())

    @classmethod
    def from_w2v_plain_format(cls, fp, delimiter=' '):
        """ Load vectors using Word2Vec CSV plain format
        """
        words = []
        vectors = []
        for line in fp.readlines():
            values = line.strip(delimiter + '\n').split(delimiter)
            word, vector = values[0], [float(x) for x in values[1:]]
            words.append(word)
            vectors.append(vector)
        return cls(words, vectors)

    @classmethod
    def from_w2v_binary_format(cls, fp, delimiter=' '):
        """ Load vectors using Word2Vec binary format
        """
        words = []
        vectors = []
        vocab_size, vec_size = map(int, fp.readline().decode('utf-8').split())
        binary_len = np.dtype('float32').itemsize * vec_size 
        for i in range(vocab_size):
            word = []
            while True:
                ch = fp.read(1)
                if ch == b'\n':
                    continue
                if ch == b' ':
                    break
                if ch == b'':
                    raise EOFError('unexpected end of input; the count may be incorrect or the file damaged')
                word.append(ch)
            word = b''.join(word).decode('utf-8')
            vector = np.fromstring(fp.read(binary_len), dtype=np.float32)
            words.append(word)
            vectors.append(vector)
        return cls(words, vectors)

    def __init__(self, words, vectors, eos=None, unk=None, pad=None):
        """ Constructor

        Arguments:
            words (list): Ordered list of words
            vectors (iterable of arrays): Vector of each words. It must have the
                same length than the word list
        """
        self.vocab = Vocab(words, eos=None, unk=None, pad=None, order_by='insertion')
        self.matrix = torch.stack([torch.Tensor(v) for v in vectors])

    @property
    def vector_size(self):
        """ Returns the vector size
        """
        return self.matrix.shape[1]

    def __len__(self):
        """ Returns the vocabulary size
        """
        return len(self.vocab)

    def __getitem__(self, item):
        """ Returns the vector of a given word or list of words

        Arguments:
            item (list, tuple or str): If list or tuple are passed it would return

        Shape:
            For a list or tuple of N element
            Output: (N, D) where D is the vector size
            otherwise
            Output: (D,) where D is the vector size
        """
        if isinstance(item, (list, tuple)):
            return self.matrix[[self.vocab[word] for word in item]]
        elif isinstance(item, str):
            return self.matrix[self.vocab[item]]
        else:
            raise ValueError("Indexing not supported to {}".format(repr(type(x))))

    def similarity(self, word_a, word_b):
        """ Returns the similarity between two words
        """
        if isinstance(word_a, list):
            vec_a = self[word_a]
        else:
            vec_a = self[word_a].unsqueeze(0)
            a_is_list = False
        if isinstance(word_b, list):
            vec_b = self[word_b]
        else:
            vec_b = self[word_b].unsqueeze(0)
        similarity = torch.cosine_similarity(vec_a, vec_b)
        if not isinstance(word_a, list) and not isinstance(word_b, list):
            return similarity.squeeze(0)
        else:
            return similarity

    def most_similar(self, positive, negative=None, topn=10):
        """ Returns the number of words in the vocabulary
        """
        if positive:
            positive = self[positive]
            if positive.ndim > 1:
                positive = positive.sum()
        else:
            positive = torch.zeros(self.vector_size)
        if negative:
            negative = self[negative]
            if negative.ndim == 1:
                negative = self[negative].sum(dim=0)
        else:
            negative = torch.zeros(self.vector_size)

        vec = (positive - negative).unsqueeze(dim=0)
        similarities_idxs = torch.cosine_similarity(vec, self.matrix)
        results = torch.topk(similarities_idxs, k=topn)
        return [(self.vocab.idx2word[i], v.item()) for i, v in zip(results.indices, results.values)]

    def __getstate__(self):
        return {'vocab': self.vocab,
                'matrix': self.matrix.tolist()}

    def __setstate__(self, d):
        self.vocab = d['words']
        self.matrix = torch.Tensor(d['matrix'])

    def _dump_as_word2vec_plain(self, fp, delimiter=' '):
        for word, vector in zip(self.vocab, self.matrix):
            vector_str = delimiter.join(map(str, vector.tolist()))
            fp.write(word)
            fp.write(delimiter)
            fp.write(vector_str)
            fp.write("\n")

    def _dump_as_word2vec_binary(self, fp):
        fp.write("{} {}\n".format(len(self), self.vector_size).encode('utf-8'))
        for word, vector in zip(self.vocab, self.matrix):
            fp.write(word.encode('utf-8'))
            fp.write(b"")
            s = struct.pack('f'*self.vector_size, *vector.tolist())
            fp.write(s)

    def _dump_as_pickle(self, fp):
        pickle.dump(self, fp)

    def save(self, path, format='plain', compressed=False):
        if format == 'plain':
            dump_fn = self._dump_as_word2vec_plain
            if compressed:
                fp = gzip.open(path, 'w+t')
            else:
                fp = open(path, 'w+')
        elif format in ('binary', 'pickle'):
            if format == 'binary':
                dump_fn = self._dump_as_word2vec_binary
            else:
                dump_fn = self._dump_as_pickle

            if compressed:
                fp = gzip.open(path, 'w+b')
            else:
                fp = open(path, 'w+b')

        with fp:
            dump_fn(fp)

    def replace_embeddings(self, vocab, embeddings, freeze=False):
        index = [self.vocab.word2idx.get(word, None) for word in vocab.idx2word]
        index_a = torch.tensor([j for j, i in enumerate(index) if i is not None])
        index_b = torch.tensor([i for i in index if i is not None])
        new_matrix = torch.index_select(self.matrix, 0, index=index_b)
        embeddings.weight.data[index_a] = new_matrix
        if freeze:
            embeddings.weight.requires_grad = False


class GLoVeVectors(KeyedVectors):
    """ Glove Vectors pretrained from
    [Wikipedia 2014](http://dumps.wikimedia.org/enwiki/20140102/) and
    [Gigaword5](https://catalog.ldc.upenn.edu/LDC2011T07)
    """
    urls = {
        "en-wiki": {
          "name": "glove.6B.zip",
          "url": "http://nlp.stanford.edu/data/glove.6B.zip",
          "files": {
            50:  "glove.6B.50d.txt",
            100: "glove.6B.100d.txt",
            200: "glove.6B.200d.txt",
            300: "glove.6B.300d.txt"
          }
        },
        "twitter": {
            "name": "glove.twitter.27B.zip",
            "url": "https://nlp.stanford.edu/data/glove.twitter.27B.zip",
            "files": {
                25: "glove.twitter.27B.25d.txt",
                50: "glove.twitter.27B.50d.txt",
                100: "glove.twitter.27B.100d.txt",
                200: "glove.twitter.27B.200d.txt",
            }
        },
        "common-crawl": {
            "name": "glove.42B.300d.zip",
            "url": "https://nlp.stanford.edu/data/glove.42B.300d.zip",
            "files": {
                300: "glove.42B.300d.txt",
            }
        },
        "common-crawl-large": {
            "name": "glove.840B.300d.zip",
            "url": "https://nlp.stanford.edu/data/glove.840B.300d.zip",
            "files": {
                300: "glove.840B.300d.txt",
            }
        },
    }

    @classmethod
    def from_url(cls, domain='en-wiki', dim=300, root=None):
        if domain not in cls.urls:
            raise ValueError("Invalid domain {}. "
                             "Available domains are: {}".format(domain,
                                                                ', '.join(map(repr, cls.urls))))
        metadata = cls.urls[domain]
        if dim not in metadata['files']:
            raise ValueError("Invalid dim {}. "
                             "Available dimensions are: {}".format(dim, ', '.join(map(str, metadata['files']))))

        dest = root or os.path.join(cls.ROOT_DIR, metadata['name'])
        if not os.path.exists(dest):
            cls.make_vectors_dir()
            download_from_url(metadata['url'], dest)

        target = metadata['files'][dim]
        with ZipFile(dest) as f:
            content = f.read(target).decode('utf-8').strip('\n')
        content = StringIO(content)
        content.seek(0)
        return cls.from_w2v_plain_format(content)

class Word2VecVectors(KeyedVectors):
    urls = {
        "GoogleNews": {
            "name": "GoogleNews-vectors-negative300.bin.gz",
            "url": "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
        }
    }

    @classmethod
    def from_url(cls, domain='GoogleNews', root=None):
        if domain not in cls.urls:
            raise ValueError("Invalid domain {}. "
                             "Available domains are: {}".format(domain,
                                                                ', '.join(map(repr, cls.urls))))
        metadata = cls.urls[domain]
        dest = root or os.path.join(cls.ROOT_DIR, metadata['name'])
        if not os.path.exists(dest):
            cls.make_vectors_dir()
            download_from_url(metadata['url'], dest)
        with gzip.open(dest, 'rb') as f:
            return cls.from_w2v_binary_format(f)
