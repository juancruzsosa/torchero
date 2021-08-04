import torch

from torchero.utils.text.transforms.compose import Compose
from torchero.utils.text.transforms.tokenizers import *
from torchero.utils.text.transforms.truncate import *
from torchero.utils.text.transforms.vocab import Vocab

tokenizers = {
    'split': lambda: str.split,
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
                         tokenizer=str.split,
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
