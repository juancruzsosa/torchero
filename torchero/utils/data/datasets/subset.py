import torch
from . import Dataset

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


class ShrinkDataset(SubsetDataset):
    """ Dataset that shrink dataset
    """
    INVALID_P_MESSAGE = "Invalid proportion number. Must lay between 0 and 1"

    def __init__(self, dataset, p=1):
        """ Constructor

        Args:
            dataset: Original dataset
            p: Shrink proportion
        """
        if p < 0 or p > 1:
            raise ValueError(self.INVALID_P_MESSAGE)

        x = round(p * len(dataset))

        if x == 0:
            indices = []
        else:
            indices = sorted(torch.randperm(len(dataset))[0:x])

        super(ShrinkDataset, self).__init__(dataset,
                                            indices)
