from collections import Counter, OrderedDict
from itertools import chain
from functools import partial

class OrderedCounter(Counter, OrderedDict):
    pass

class OutOfVocabularyError(LookupError):
    pass

def _add_special_tokens(tokens, bos=None, eos=None):
    tokens = list(tokens)
    if bos is not None:
        tokens.insert(0, bos)
    if eos is not None:
        tokens.append(eos)
    return tokens

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
    ORDER_BY_OPTIONS = ('frequency', 'insertion', 'alpha')
    INVALID_ORDER_BY_ARGUMENT_MESSAGE = ("Invalid argument for order_by. "
                                         "Use {options}, or a function. "
                                         "Got: {value}")

    @classmethod
    def build_from_texts(cls, texts, bos=None, eos=None, pad=None, unk=None, max_size=None, min_count=1):
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
        examples = chain.from_iterable(map(partial(_add_special_tokens, eos=eos, bos=bos), texts))
        vocab = cls(examples, eos=eos, bos=bos, pad=pad, unk=unk, max_size=max_size, min_count=min_count)
        return vocab

    def __init__(self, vocab={}, bos=None, eos=None, pad=None, unk=None, max_size=None, order_by='frequency', min_count=1):
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
            order_by_freq (bool): If true the order of insertion is ordering from frequency
        """
        self.pad = pad
        self.eos = eos
        self.bos = bos
        self.unk = unk
        self.max_size = max_size
        self.min_count = min_count
        self.freq = Counter()
        self.idx2word = list()
        self.word2idx = {}
        self.start_index = 0
        if order_by is None:
            raise ValueError(self.INVALID_ORDER_BY_ARGUMENT_MESSAGE.format(
                             options=self.ORDER_BY_OPTIONS, value=order_by))
        self.default_order = order_by
        if self.pad is not None: # Padding always go to index 
            self.add([self.pad])
        if unk is not None:
            self.add([self.unk])
        if bos is not None:
            self.add([self.bos])
        if eos is not None:
            self.add([self.eos])
        self.add(vocab, order_by=order_by, min_count=self.min_count)

    def add(self, vocab, order_by=None, min_count=None):
        """ Merge vocabularies. 

        Note:
            *) The order for numbering new words uses the `order_by` parameter.
                By default it orders from the most frequent to the least frequent.
            *) The number of new added terms depends on the vocabulary max size
                setted at vocabulary construction time.
            *) This method preserves the index numbers for the old words.

        Arguments:
            vocab (iterable or mapping): If a mapping is passed it will be used
                as a dictionary of terms counts. If a iterable is passed the
                elements are counted from the iterable. Default: {}
            order_by (None or str or function of tuples): If none is passed used 
                the order passed at creation. If 'insertion'
                is passed the vocab iteration order is used, if 'frequency'
                is passed the frequency order is used, if 'alpha'
                is passed the order of the keys will be used. If a function
                is passed the function is used to sort the elements by 
                applying it for each tuple (word, frequency). Default: 'freq'
        """
        if (isinstance(order_by, str) and order_by not in self.ORDER_BY_OPTIONS) or \
           (order_by is not None and
            not isinstance(order_by, str) and
            not callable(order_by)):
                raise ValueError(self.INVALID_ORDER_BY_ARGUMENT_MESSAGE.format(
                                    options=self.ORDER_BY_OPTIONS,
                                    value=order_by))
        vocab = OrderedCounter(vocab)

        if order_by is None: # Insertion order
            order_by = self.default_order
        if isinstance(order_by, str):
            if order_by == 'frequency': # Use frequency ordering
                it = vocab.most_common()
            elif order_by == 'insertion': # Use insertion Ordering
                it = vocab.items()
            else: # Alpha. Use alphabetical order
                it = sorted(vocab.items(), key=lambda x: x[0])
        else: # Use function order
            it = sorted(vocab.items(), key=order_by)

        for word, freq in it:
            if word in self.word2idx: # The word is already in the vocabulary
                self.freq[word] += freq
            elif (self.max_size is None or len(self) < self.max_size) and (min_count is None or freq >= min_count):
                self.freq[word] = freq
                self.word2idx[word] = len(self)
                self.idx2word.append(word)

    def __len__(self):
        """ Returns the number of terms in the vocabulary
        """
        return len(self.word2idx)

    def __iter__(self):
        """ Returns an iterator to word in the vocabulary
        """
        return iter(self.idx2word)

    def __contains__(self, word):
        """ vocab.__contains__(word) <==> word in vocab
        """
        return word in self.word2idx

    def words(self):
        return iter(self)

    def indexes(self):
        yield from range(self.start_index, len(self)+1)

    def items(self):
        """ Returns an iterator to
        """
        yield from ((word, self.start_index + i) for i, word in enumerate(self.idx2word))

    def __getitem__(self, word):
        """ Returns the index of the given term
        """
        index = self.word2idx.get(word, self.word2idx[self.unk] if self.unk else None)
        if index is None:
            raise OutOfVocabularyError("{} is not in the Vocabulary".format(word))
        else:
            return index + self.start_index

    def __repr__(self):
        opt_params = {'bos': self.bos,
                      'eos': self.eos,
                      'pad': self.pad,
                      'unk': self.unk,
                      'max_size': self.max_size,
                      'order_by': self.default_order,
                      'min_count': self.min_count}
        format_params = ',\n\t'.join(
            '{}={}'.format(name, repr(param))
            for name, param in opt_params.items()
            if param is not None
        )
        return '{}({})'.format(self.__class__.__name__,
                               format_params)

    def __call__(self, tokens):
        """ Converts list of tokens to list of indexes
        """
        ids_seq = []
        if self.bos is not None:
            ids_seq.append(self[self.bos])
        if self.unk is None:
            ids_seq.extend([self[token] for token in tokens if token in self.word2idx])
        else:
            ids_seq.extend([self[token] for token in tokens])
        if self.eos is not None:
            ids_seq.append(self[self.eos])
        return ids_seq


    def __getstate__(self):
        return {'words': self.idx2word,
                'freqs': [self.freq[word] for word in self.idx2word],
                'pad': self.pad,
                'eos': self.eos,
                'bos': self.bos,
                'unk': self.unk,
                'max_size': self.max_size,
                'start_index': self.start_index,
                'min_count': self.min_count,
                'default_order': self.default_order
                }

    def __setstate__(self, d):
        self.start_index = d['start_index']
        self.idx2word = d['words']
        self.word2idx = {word: i for i, word in enumerate(d['words'])}
        self.freq = Counter(d['freqs'])
        self.pad = d['pad']
        self.eos = d['eos']
        self.bos = d['bos']
        self.unk = d['unk']
        self.max_size = d['max_size']
        self.min_count = d['min_count']
        self.default_order = d['default_order']

    def fit(self, texts):
        """ Updates the Vocabulary from a list of tokenized sentences

        Arguments:
            texts (list[list[str]]): Corpus (already tokenized) used to build the vocabulary
        """
        examples = chain.from_iterable(map(partial(_add_special_tokens, eos=self.eos, bos=self.bos), texts))
        self.add(examples)
