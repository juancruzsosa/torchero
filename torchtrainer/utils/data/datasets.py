import torch
from torch.utils.data import Dataset

class SubsetDataset(Dataset):
    """ Dataset that is a subset from another Dataset
    """

    def __init__(self, dataset, indices):
        """ Constructor

        Args:
            dataset: Original dataset
            indices: Indices of original dataset for sample
        """
        self._dataset = dataset
        self._indices = indices

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        return self._dataset[self._indices[idx]]


class UnsuperviseDataset(Dataset):
    """ Supervised to Unsupervised Dataset adapter
    """

    def __init__(self, dataset, input_indices=[0]):
        """ Constructor

        Arguments:
            dataset: Original dataset
        """
        self._dataset = dataset
        self._indices = input_indices

    def __getitem__(self, idx):
        element = self._dataset[idx]

        if len(self._indices) == 1:
            return element[self._indices[0]]
        else:
            return tuple(element[i] for i in self._indices)

    def __len__(self):
        return len(self._dataset)
