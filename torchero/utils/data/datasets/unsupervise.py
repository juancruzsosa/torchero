from torchero.utils.data.datasets import Dataset


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

    def __getattr__(self, attr):
        return getattr(self._dataset, attr)

    def __str__(self):
        return str(self._dataset)

    def __repr__(self):
        return "{} of \n {}".format(self.__class__.__name__,
                                    repr(self._dataset))
