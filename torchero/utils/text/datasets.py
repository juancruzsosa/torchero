import csv
import json

import torch
from torch.utils.data import Dataset, DataLoader

from torchero.utils.collate import PadSequenceCollate
from torchero.utils.text.tokenizers import tokenizers
from torchero.utils.text.vocab import Vocab


class TextClassificationDataset(Dataset):
    @classmethod
    def from_json(
        cls,
        path,
        text_col,
        target_col,
        tokenizer=str.split,
        vocab=None,
        vocab_max_size=None,
        vocab_min_count=1,
        eos=None,
        pad="<pad>",
        unk="<unk>",
        transform=str.lower,
        transform_target=None,
    ):
        with open(path, 'r') as jsonfile:
            records = json.load(jsonfile)
            texts = [r[text_col] for r in records]
            targets = [r[target_col] for r in records]
            return cls(
                texts,
                targets,
                tokenizer=tokenizer,
                vocab=vocab,
                vocab_max_size=vocab_max_size,
                vocab_min_count=vocab_min_count,
                eos=eos,
                pad=pad,
                unk=unk,
                transform=transform,
                transform_target=transform_target,
            )

    @classmethod
    def from_csv(
        cls,
        path,
        text_col,
        target_col,
        delimiter=",",
        quotechar='"',
        has_header=True,
        column_names=None,
        tokenizer=str.split,
        vocab=None,
        vocab_max_size=None,
        vocab_min_count=1,
        eos=None,
        pad="<pad>",
        unk="<unk>",
        transform=str.lower,
        transform_target=None,
    ):
        def check_column(col, columns):
            if not isinstance(col, (int, str)):
                raise TypeError("invalid column type")
            if isinstance(col, str) and col not in columns:
                raise RuntimeError(
                    "text column {} not found in csv columns".format(
                        repr(text_col)
                    )
                )
            elif isinstance(col, int) and col >= len(columns):
                raise RuntimeError(
                    "text column index is {} but the csv has only {} columns".format(
                        text_col, len(columns)
                    )
                )

        with open(path, newline="") as csvfile:
            recordsreader = csv.reader(
                csvfile, delimiter=delimiter, quotechar=quotechar
            )
            it = iter(recordsreader)
            if has_header:
                csv_col_names = next(it)
                column_names = list(column_names or csv_col_names)
            if column_names is not None:
                check_column(text_col, column_names)
                check_column(target_col, column_names)
            if isinstance(text_col, str):
                text_col = column_names.index(text_col)
            if isinstance(target_col, str):
                target_col = column_names.index(target_col)
            texts = []
            targets = []
            for row in it:
                texts.append(row[text_col])
                targets.append(row[target_col])
            return cls(
                texts,
                targets,
                tokenizer=tokenizer,
                vocab=vocab,
                vocab_max_size=vocab_max_size,
                vocab_min_count=vocab_min_count,
                eos=eos,
                pad=pad,
                unk=unk,
                transform=transform,
                transform_target=transform_target,
            )

    def __init__(
        self,
        texts,
        targets,
        tokenizer=str.split,
        vocab=None,
        vocab_max_size=None,
        vocab_min_count=1,
        eos=None,
        pad="<pad>",
        unk="<unk>",
        transform=str.lower,
        transform_target=None,
    ):
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
        self.vocab = vocab or Vocab.build_from_texts(
            samples, eos=eos, pad=pad, unk=unk, max_size=vocab_max_size, min_count=vocab_min_count
        )

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
