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
