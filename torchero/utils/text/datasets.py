import csv
import json

import torch
from torch.utils.data import Dataset, DataLoader

from torchero.utils.collate import PadSequenceCollate


class LabeledTextDataset(Dataset):
    """ Dataset for labeled text
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
        transform_text,
        transform_target=_default_transform_target):
        """ Creates a class instance from a json file (with list of dict fields scheme).

        Arguments:
            path (str or Path): Path of the json file
            text_col (str or int): If a string is passed the column name for the
                texts. If a int is passed the index of the column for the text
            text_col (str, int, list of str or list of str): If a string is passed the column name for the
                target. If a int is passed the index of the column for the target
            transform (TextTransform): See class instance constructor
            transform_target (callable): See class instance constructor
        """
        with open(path, 'r') as jsonfile:
            records = json.load(jsonfile)
            texts = [r[text_col] for r in records]
            targets = [self.get_record_item(record, target_col) for r in records]
            return cls(
                texts,
                targets,
                transform_text=transform_text,
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
        transform_text=str.lower,
        transform_target=torch.tensor,
    ):
        """ Creates a class instance from a csv file

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
            transform_text (TextTransform): Function used for text preprocessing before to perform tokenization
            transform_target (callable): Function used to transform targets.
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
                transform_text=transform_text,
                transform_target=transform_target,
            )

    def __init__(
        self,
        texts,
        targets,
        transform_text,
        transform_target=torch.tensor,
        build_vocab=True,
    ):
        """ Constructor

        Arguments:
            texts (list-like of str): List of dataset text samples
            targets (list-like): List of targets for every sample in texts.
            transform_text (TextTransform): Function used for text preprocessing before to
                perform tokenization
            transform_target (callable): Function used to transform targets.
        """
        if len(texts) != len(targets):
            raise RuntimeError(
                "The number of texts should equal the number of targets"
            )
        self.texts = texts
        self.targets = targets
        self.transform_text = transform_text
        self.transform_target = transform_target
        if build_vocab:
            self.transform_text.build_vocab(self.texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        text = self.texts[i]
        ids = self.transform_text(text)
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

class TextClassificationDataset(LabeledTextDataset):
    """ Dataset for Text classification task
    """
    pass
