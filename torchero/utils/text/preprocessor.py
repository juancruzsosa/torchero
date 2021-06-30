import torch

from torchero.utils.text.tokenizers import tokenizers
from torchero.utils.text.vocab import Vocab

class TextTransform(object):
    def __init__(self,
                 pre_process=str.lower,
                 tokenizer=str.split,
                 vocab=None,
                 vocab_max_size=None,
                 vocab_min_count=1,
                 eos=None,
                 pad="<pad>",
                 unk=None,
                 post_process=torch.LongTensor):
        self.pre_process = pre_process
        if isinstance(tokenizer, str):
            tokenizer = tokenizers[tokenizer]
        self.tokenizer = tokenizer
        if vocab is None:
            vocab = Vocab(eos=eos,
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
        ids = self.vocab(tokens)
        ids = self.post_process(ids)
        return ids

    def build_vocab(self, samples):
        samples = map(self.pre_process, samples)
        tokenized_samples = map(self.tokenizer, samples)
        self.vocab.add_from_tokenized_texts(tokenized_samples)
