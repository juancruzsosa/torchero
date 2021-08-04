import torch

from torchero.utils.text.tokenizers import tokenizers
from torchero.utils.text.vocab import Vocab
from torchero.utils.text.truncate import LeftTruncator, RightTruncator, CenterTruncator

class TextTransform(object):
    def __init__(self,
                 pre_process=str.lower,
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
        self.pre_process = pre_process
        if isinstance(tokenizer, str):
            tokenizer = tokenizers[tokenizer]()
        self.tokenizer = tokenizer
        if max_len is not None:
            if truncate_mode == 'left':
                self.truncator = LeftTruncator(max_len)
            elif truncate_mode == 'right':
                self.truncator = RightTruncator(max_len)
            elif truncate_mode == 'center':
                self.truncator = CenterTruncator(max_len)
            else:
                raise ValueError("Invalid truncate_mode. Choose from left, right, center")
        else:
            self.truncator = None
        if vocab is None:
            vocab = Vocab(bos=bos,
                  eos=eos,
                  pad=pad,
                  unk=unk,
                  max_size=vocab_max_size,
                  order_by='frequency',
                  min_count=vocab_min_count)
        self.vocab = vocab
        self.post_process = post_process

    def __call__(self, text):
        text = self.pre_process(text)
        tokens = self.tokenizer(text)
        if self.truncator is not None:
            tokens = self.truncator(tokens)
        ids = self.vocab(tokens)
        ids = self.post_process(ids)
        return ids

    def build_vocab(self, samples):
        samples = map(self.pre_process, samples)
        tokenized_samples = map(self.tokenizer, samples)
        self.vocab.add_from_tokenized_texts(tokenized_samples)
