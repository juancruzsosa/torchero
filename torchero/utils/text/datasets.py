import torch
from torch.utils.data import Dataset, DataLoader

from torchero.utils.collate import PadSequenceCollate
from torchero.utils.text.tokenizers import tokenizers
from torchero.utils.text.vocab import Vocab

class TextClassificationDataset(Dataset):
    def __init__(self, texts, targets, tokenizer=str.split, vocab=None, vocab_max_size=None, eos=None, pad='<pad>', unk='<unk>', transform=str.lower, transform_target=None):
        if len(texts) != len(targets):
            raise RuntimeError("The number of texts should equal the number of targets")
        self.texts = texts
        self.targets = targets
        self.transform = transform
        self.transform_target = transform_target
        if isinstance(tokenizer, str):
            tokenizer = tokenizers[tokenizer]
        self.tokenizer = tokenizer

        samples = self.texts
        samples = map(self.transform, samples)
        samples = map(self.tokenizer, samples)
        self.vocab = vocab or Vocab.build_from_texts(samples, eos=eos, pad=pad, unk=unk, max_size=vocab_max_size)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        text = self.texts[i]
        text = self.transform(text)
        text = self.tokenizer(text)
        ids = self.vocab(text)
        ids = torch.LongTensor(ids)
        target = self.targets[i]
        if self.transform_target:
            target = self.transform_target(target)
        return ids, target

    def dataloader(self, *args, **kwargs):
        kwargs['collate_fn'] = kwargs.get('collate_fn', PadSequenceCollate(pad_value=self.vocab[self.vocab.pad]))
        return DataLoader(self, *args, **kwargs)
