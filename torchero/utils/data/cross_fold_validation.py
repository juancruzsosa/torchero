import math

import torch

from torchero.utils.data.datasets import SubsetDataset


class CrossFoldValidation(object):
    """
    Iterator through cross-fold-validation folds
    """

    INVALID_VALID_SIZE_MESSAGE = (
        'Invalid `valid_size`: `valid_size` must lay between 0.0 and 1.0'
    )
    TRAIN_AND_VALID_DATASET_SIZE_MESSAGE = (
        'Train and valid dataset must have same size'
    )

    def __init__(self, dataset, valid_dataset=None, valid_size=0.2):
        """ Constructor

        Args:
            dataset (`torch.utils.data.Dataset`):
                Dataset to split
            valid_dataset (`torch.utils.data.Dataset`):
                Validation dataset where to take validation samples. Should be
                the same dataset as the training dataset but possibly with
                different transformations (for example training dataset is
                augmented and validation is not). If None the `dataset'
                argument its used (default).
            valid_size (float):
                Proportion of datset destined to validation.
                Must lay between 0 and 1
        """

        valid_size = round(len(dataset) * valid_size)
        if valid_size <= 0 or valid_size >= len(dataset):
            raise Exception(self.INVALID_VALID_SIZE_MESSAGE)

        if valid_dataset is not None and len(valid_dataset) != len(dataset):
            raise Exception(self.TRAIN_AND_VALID_DATASET_SIZE_MESSAGE)

        self.valid_size = valid_size
        self.dataset = dataset
        self.valid_dataset = valid_dataset or dataset
        self.indices = list(torch.randperm(len(dataset)))

    def __len__(self):
        return math.ceil(len(self.dataset) / self.valid_size)

    def __iter__(self):
        indices = self.indices

        for valid_start in range(0, len(self.dataset), self.valid_size):
            valid_indices = indices[valid_start: valid_start + self.valid_size]

            train_indices = (indices[:valid_start]
                             + indices[valid_start + self.valid_size:])

            train_dataset = SubsetDataset(dataset=self.dataset,
                                          indices=train_indices)
            valid_dataset = SubsetDataset(dataset=self.valid_dataset,
                                          indices=valid_indices)

            yield train_dataset, valid_dataset


def train_test_split(dataset, valid_size=0.2):
    cv = CrossFoldValidation(dataset, valid_size=valid_size)
    train_dataset, valid_dataset = next(iter(cv))
    return train_dataset, valid_dataset
