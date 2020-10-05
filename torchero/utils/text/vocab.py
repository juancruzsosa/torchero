from collections import Counter
from itertools import chain

class Vocab(object):
    """ Represents a Vocabulary to map objects to ids

    Example:
        >>> v1 = Vocab({'the': 5, 'of': 1, 'and': 3})
        >>> v1
             {'<pad>': 0, '<unk>': 1, 'the': 2, 'and': 3, 'of': 4}
        >>> v1['and']
            3
        >>> v1['xyz']
            1
        >>> v1.freq['and']
            3
        >>> v2 = Vocab(['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'other', 'fox'])
        >>> v2
            {'<pad>': 0, '<unk>': 1, 'the': 2, 'fox': 3, 'quick': 4, 'brown': 5, 'jumps': 6, 'over': 7, 'other': 8}
    """
    @classmethod
    def build_from_texts(cls, texts, eos=None, pad='<pad>', unk='<unk>', max_size=None):
        """ Builds a Vocabulary from a list of sentences

        Arguments:
            texts (list[list[str]]): Corpus used to build the vocabulary
            eos (str): Optional special token for the end of each sentence.
            pad (str): Special token for strings padding
            unk (str): Special token for Out of Vocabulary terms.
            max_size (int): Maximum size of the Vocabulary in number of
                elements. The discarded elements are selected from the list of less frequent terms.
                If None is passed the size of the vocabulary will have no growth limit.
        """
        if eos is None:
            examples = chain.from_iterable(texts)
        else:
            examples = chain.from_iterable(map(lambda x: x + [eos]), texts)
        vocab = cls(examples, eos=eos, pad=pad, unk=unk, max_size=max_size)
        return vocab

    def __init__(self, vocab={}, eos=None, pad='<pad>', unk='<unk>', max_size=None):
        """ Constructor

        Arguments:
            vocab (iterable or mapping): If a mapping is passed it will be used
                as a dictionary of terms counts. If a iterable is passed the
                elements are counted from the iterable. Default: {}
            eos (str): Optional special token for the end of each sentence.
            pad (str): Special token for strings padding
            unk (str): Special token for Out of Vocabulary terms.
            max_size (int): Maximum size of the Vocabulary in number of
                elements. The discarded elements are selected from the list of less frequent terms.
                If None is passed the size of the vocabulary will have no growth limit.
        """
        self.pad = pad
        self.eos = eos
        self.unk = unk
        self.max_size = max_size
        self.freq = Counter()
        self.idx2word = list()
        self.word2idx = {}
        if self.pad is not None:
            self.last_index = -1
            self.add([self.pad])
        else:
            self.last_index = 0
        if unk is not None:
            self.add([self.unk])
        if eos is not None:
            self.add([self.eos])
        self.add(vocab)

    def add(self, vocab):
        """ Merge vocabularies

        Arguments:
            vocab (iterable or mapping): If a mapping is passed it will be used
                as a dictionary of terms counts. If a iterable is passed the
                elements are counted from the iterable. Default: {}
        """
        new_vocab = Counter(vocab)
        for word, freq in new_vocab.most_common():
            if word in self.word2idx:
                self.freq[word] += freq
            elif self.max_size is None or self.last_index < self.max_size:
                self.freq[word] += freq
                self.last_index += 1
                self.word2idx[word] = self.last_index
                self.idx2word.append(word)
            else:
                break

    def __len__(self):
        """ Returns the number of terms in the vocabulary
        """
        return len(self.word2idx)

    def __iter__(self):
        """ Returns an iterator to word indexes
        """
        return iter(self.word2idx)

    def __getitem__(self, word):
        """ Returns the index of the given term
        """
        return self.word2idx.get(word, self.word2idx[self.unk] if self.unk else None)

    def __repr__(self):
        return repr(dict(self.word2idx))

    def __call__(self, tokens):
        """ Converts list of tokens to list of indexes
        """
        ids_seq = [self[token] for token in tokens]
        if self.unk is None:
            ids_seq = [idx for idx in ids_seq if idx is not None]
        if self.eos is  not None:
            ids_seq.append(self.word2idx[self.eos])
        return ids_seq

    def __getstate__(self):
        return {'words': self.idx2word,
                'freqs': [self.freq[word] in self.idx2word],
                'pad': self.pad,
                'eos': self.eos,
                'unk': self.unk,
                'max_size': self.max_size,
                'start_index': self.start_index
                }

    def __setstate__(self, d):
        self.start_index = d['start_index']
        self.idx2word = d['words']
        self.word2idx = {word: i for i, word in enumerate(d['words'])}
        self.freq = Counter(d['freqs'])
        self.pad = d['pad']
        self.eos = d['eos']
        self.unk = d['unk']
        self.max_size = d['max_size']
