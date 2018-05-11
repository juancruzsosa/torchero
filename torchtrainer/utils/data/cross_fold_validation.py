import torch
from .datasets import SubsetDataset
import math

class CrossFoldValidation(object):
    """
    Iterator through cross-fold-validation folds
    """

    INVALID_VALID_SIZE_MESSAGE = ('Invalid `valid_size`: `valid_size` must lay '
                                  'between 0.0 and 1.0')

    def __init__(self, dataset, valid_size=0.2):
        """ Constructor

        Args:
            dataset (`torch.utils.data.Dataset`):
                Dataset to split
            valid_size (float):
                Proportion of datset destined to validation.
                Must lay between 0 and 1
        """

        valid_size = round(len(dataset) * valid_size)
        if valid_size <= 0 or valid_size >= len(dataset):
            raise Exception(self.INVALID_VALID_SIZE_MESSAGE)

        self.valid_size = valid_size
        self.dataset = dataset
        self.indices = list(torch.randperm(len(dataset)))

    def __len__(self):
        return math.ceil(len(self.dataset) / self.valid_size)

    def __iter__(self):
        for valid_start in range(0, len(self.dataset), self.valid_size):
            valid_indices = self.indices[valid_start:
                                         valid_start + self.valid_size]

            train_indices = (self.indices[:valid_start]
                             + self.indices[valid_start + self.valid_size:])

            train_dataset = SubsetDataset(dataset=self.dataset,
                                          indices=train_indices)
            valid_dataset = SubsetDataset(dataset=self.dataset,
                                          indices=valid_indices)

            yield train_dataset, valid_dataset
