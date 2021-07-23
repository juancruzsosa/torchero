import csv
import json

import torch

from torchero.utils.collate import PadSequenceCollate
from torchero.utils.data.datasets import TabularDataset

class TextClassificationDataset(TabularDataset):
    """ Dataset for text classification tasks. It can be instanciated
    from a list of examples and a list of targets, a pandas DataFrame, a csv and a json.

    Example:
        >>> t = TextTransform()
        >>> ds = TextClassificationDataset(['This movie is nice', 'This movie sucks', 'I don\'t have anything to said'],
                                           ['positive', 'negative', 'neutral'],
                                           transform=t)
        >>> t.build_vocab(ds.texts())
        >>> ds[0]
    """


    def __init__(self,
                 texts,
                 targets,
                 transform=None,
                 target_transform=None,
                 squeeze_data=True,
                 squeeze_targets=True):
        """ Constructor

        Arguments:
            texts (iter-like, pd.DataFrame, pd.Series): Iterable of texts inputs.
                If a DataFrame is passed it should have one single column
            targets (iter-like, pd.DataFrame, pd.Series): List of target for each input.
            transform (``torchero.utils.text.TextTransform``, optional): TextTransform
                instance to convert texts to tensors
            target_transform (``torchero.utils.text.target_transform``, optional): Function
                to be applied to every target
            squeeze_data (bool): When set to True if the text data is a single column every item of the
                dataset returns a string instead of a vector.
            squeeze_targets (bool): When set to True if the target is a single column every item of the
                dataset returns a scalar instead of a vector.
        """
        super(TextClassificationDataset, self).__init__(texts,
                                                        targets,
                                                        transform=transform,
                                                        target_transform=target_transform or torch.tensor,
                                                        squeeze_data=squeeze_data,
                                                        squeeze_targets=squeeze_targets)
        if len(self.data.columns) > 1:
            raise TypeError("Only one text column is allowed")
        self.target_data = self.target_data.astype('category')
        self.classes = self.target_data.apply(lambda x: x.cat.categories).to_dict()
        if len(self.target_data.columns) > 1 and any(len(t) > 2 for t in self.classes.values()):
            raise ValueError("Targets both multiclass & multilabel is not supported yet!")
        self.target_data = self.target_data.apply(lambda x: x.cat.codes)

    @property
    def idx2cat(self):
        """ Returns the classes
        """
        return self.classes

    def texts(self):
        """ Returns an iterator of the text column
        """
        yield from self.data[self.data.columns[0]].tolist()
