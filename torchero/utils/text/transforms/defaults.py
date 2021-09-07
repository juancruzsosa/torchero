import torch

from torchero.utils.text.transforms.compose import Compose
from torchero.utils.text.transforms.tokenizers import *
from torchero.utils.text.transforms.truncate import *
from torchero.utils.text.transforms.vocab import Vocab

tokenizers = {
    'basic': BasicTokenizer,
    'split': WhitespaceTokenizer,
    'spacy': SpacyTokenizer,
    'spacy:en': EnglishSpacyTokenizer,
    'spacy:es': SpanishSpacyTokenizer,
    'spacy:fr': FrenchSpacyTokenizer,
    'spacy:de': GermanSpacyTokenizer,
    'nltk': NLTKWordTokenizer,
    'nltk:word': NLTKWordTokenizer,
    'nltk:tweet': NLTKTweetTokenizer
}

truncators = {
    'left': LeftTruncator,
    'right': RightTruncator,
    'center': CenterTruncator
}

def basic_text_transform(pre_process=str.lower,
                         tokenizer='basic',
                         vocab=None,
                         vocab_max_size=None,
                         vocab_min_count=1,
                         max_len=None,
                         truncate_mode='left',
                         bos=None,
                         eos=None,
                         pad="<pad>",
                         unk=None,
                         post_process=torch.LongTensor):
    """ Build a basic pipeline for Text preprocessing

    Arguments:
        pre_process (callable, optional): Preprocessing function (e.g. to
            lower). If None is passed no string preprocessing will be done
        tokenizer (str or callable, optional): Tokenization method. If a string
            is passed it shoud be one of {tokenizers}. If a callable is passed
            the function should take a string and return the list of tokenized words (not ids).
        vocab (``torchero.utils.text.transforms.Vocab``, optional): Vocabulary
            to use. If not passed it will be built from created from
            vocab_max_size, vocab_min_count.
        vocab_max_size (int, optional): Maximum size of the Vocabulary in
            number of elements. The discarded elements are selected from the list
            of less frequent terms. If None is passed the size of the vocabulary
            will have no growth limit. Only set this argument
            if vocab argument is setted to None.
        vocab_min_count (int): Minimum frequency for a word to be
            added to the vocabulary. Only set this argument
            if vocab argument is setted to None.
        max_len (int, optional): Maximum number of tokens per sample. Only set this
        truncate_mode (str, ``torchero.utils.text.transforms.Truncate``):
            Truncation strategy. If a string is passed it should be one of
            {truncators}. If a callable is passed it should take the list of tokens
            and returns the list truncated
        bos (str, optional): Special token for begining of sentence (it will be added
            at the beginning of each text)
        eos (str, optional): Special token for ending of sentence (it will be added at
            the end of each text)
        unk (str, optional): Special token to replace each out of vocabulary
            word
        post_process (callable, optional): Last processing step. Usually
            convert the list of token ids to a torch.LongTensor
    """.format(tokenizers=', '.join(map(repr, tokenizers.keys())),
               truncators=', '.join(map(repr, truncators.keys())))
    pipeline = {}
    if pre_process is not None:
        pipeline['pre_process'] = pre_process
    if isinstance(tokenizer, str):
        tokenizer = tokenizers[tokenizer]()
    pipeline['tokenize'] = tokenizer
    if max_len is not None:
        if isinstance(truncate_mode, str):
            if truncate_mode not in truncators.keys():
                raise ValueError("Invalid truncate_mode. Choose from left, right, center")
            truncator = truncators[truncate_mode](max_len)
        else:
            truncator = truncate_mode
        pipeline['truncate'] = truncator
    if vocab is None:
        vocab = Vocab(bos=bos,
              eos=eos,
              pad=pad,
              unk=unk,
              max_size=vocab_max_size,
              order_by='frequency',
              min_count=vocab_min_count)
    pipeline['vocab'] = vocab
    if post_process is not None:
        pipeline['post_process'] = post_process

    return Compose.from_dict(pipeline)
