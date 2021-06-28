import csv
import json

import torch
from torch.utils.data import Dataset, DataLoader

from torchero.utils.collate import PadSequenceCollate
from torchero.utils.text.tokenizers import tokenizers
from torchero.utils.text.vocab import Vocab


class TextClassificationDataset(Dataset):
    """ Dataset for Text classification task
    """
    @staticmethod
    def get_record_item(record, target_field):
        if isinstance(target_field, (str, int)):
            return record[target_field]
        elif isinstance(target_field, list):
            return [record[field] for field in target_field]
        else:
            raise TypeError("Invalid `target_field` type")
    
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
        transform_target=None):
        """ Creates a TextClassificationDataset from a json file (with list of dict fields scheme).

        Arguments:
            path (str or Path): Path of the json file
            text_col (str or int): If a string is passed the column name for the
                texts. If a int is passed the index of the column for the text
            text_col (str, int, list of str or list of str): If a string is passed the column name for the
                target. If a int is passed the index of the column for the target
            tokenizer (callable): See TextClassificationDataset constructor
            vocab (`torchero.utils.text.Vocab`): See TextClassificationDataset constructor
            vocab_min_count (int): See TextClassificationDataset constructor
            eos (str): See TextClassificationDataset constructor
            pad (str): See TextClassificationDataset constructor
            unk (str): See TextClassificationDataset constructor
            transform (callable): See TextClassificationDataset constructor
            transform_target (callable): See TextClassificationDataset constructor
        """
        with open(path, 'r') as jsonfile:
            records = json.load(jsonfile)
            texts = [r[text_col] for r in records]
            targets = [self.get_record_item(record, target_col) for r in records]
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
        """ Creates a TextClassificationDataset from a csv file

        Arguments:
            path (str or Path): Path of the csv file
            text_col (str or int): If a string is passed the column name for the
                texts. If a int is passed the index of the column for the text
            text_col (str or int): If a string is passed the column name for the
                target. If a int is passed the index of the column for the target
            delimiter (str): Character used to splits the csv fields
            quotechar (str): Character used to delimit the text strings
            has_header (bool): True if the csv contains a header. False, otherwise
            column_names (list): List of columns names
            tokenizer (callable): See TextClassificationDataset constructor
            vocab (`torchero.utils.text.Vocab`): See TextClassificationDataset constructor
            vocab_min_count (int): See TextClassificationDataset constructor
            eos (str): See TextClassificationDataset constructor
            pad (str): See TextClassificationDataset constructor
            unk (str): See TextClassificationDataset constructor
            transform (callable): See TextClassificationDataset constructor
            transform_target (callable): See TextClassificationDataset constructor
        """
        def check_column(col, columns, single=True):
            permitted_types = (int, str) if single else (int, str, list)
            if not isinstance(col, permitted_types):
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
                check_column(target_col, column_names, single=False)
            if isinstance(text_col, str):
                text_col = column_names.index(text_col)
            if isinstance(target_col, str):
                target_col = column_names.index(target_col)
            else: # isinstance(target_col, list)
                target_col = [column_names.index(t) for t in target_col]
            texts = []
            targets = []
            for row in it:
                texts.append(row[text_col])
                targets.append(cls.get_record_item(row, target_col))
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
        """ Constructor

        Arguments:
            texts (list-like of str): List of dataset text samples
            targets (list-like): List of targets for every sample in texts.
            tokenizer (callable): Function used to tokenize the text.
            vocab (`torchero.utils.text.Vocab`): Corpus Vocabulary. If None is passed
            the vocabulary is built from scratch from the corpus first.
            vocab_max (int): Limits the created vocabulary to have less than `vocab_max` tokens.
                If None the created vocabulary will have no limit.
                Only valid when vocab argument is None. Default: None
            vocab_min_count (int): Minimum number of occurences for a token to
                be part of the vocabulary. Only valid when vocab argument is None.
                Default: None
            eos (str): Special token to be used for the end of every sentences.
            pad (str): Special token to be used for padding.
            unk (str): Special token to be used for unknown words.
                If unk is None, unknown words will be skipped from the corpus.
                Default: '<unk>'
            transform (callable): Function used for text preprocessing before to
                perform tokenization
            transform_target (callable): Function used to transform targets.
        """
        if len(texts) != len(targets):
            raise RuntimeError(
                "The number of texts should equal the number of targets"
            )
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
        kwargs["collate_fn"] = kwargs.get(
            "collate_fn",
            PadSequenceCollate(pad_value=self.vocab[self.vocab.pad]),
        )
        return DataLoader(self, *args, **kwargs)
