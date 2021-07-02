import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


class PadSequenceCollate(object):
    """ Collator for sequence padding. Useful for handle batchs of variable size tensors

    Converts list of tuples ($x_i$, $y_i$) where $x_i$ is a tensor with shape (N_i, *)
    into a tuple (X, L), Y where X is a single padded.

    Example:
        >>> collate =  PadSequenceCollate()
        >>> X_1, y_1 = torch.tensor([1, 2]), torch.tensor(1)
        >>> X_2, y_2 = torch.tensor([3]), torch.tensor(0)
        >>> collate([(X_1, y_1), (X_2, y_2)])
        (tensor([[1, 2], [0, 3]), torch([2, 1])), tensor([1., 0.])
        >>> collate =  PadSequenceCollate(pad_value=-1, padding_scheme='right')
        (tensor([[1, 2], [3, -1]), torch([2, 1])), tensor([1., 0.])

    This is ment to be used in the set-up of a DataLoader as collate_fn parameter
    """

    def __init__(self, pad_value=0, padding_scheme='right', ret_lengths=True, pad_dims=(0,)):
        """ Constructor

        Arguments:
            pad_value: Value used for fill the padding
            padding_scheme: Padding scheme. One of 'left', 'right', 'center'. If
                'left' is passed the padding is added at the begining of the
                sequence. If 'right' is passed the padding is added to the end.
                If 'center' is passed the half of the padding is added at the begining
                and half at the end of the sequence
        """
        self.pad_value = pad_value
        if padding_scheme not in ('left', 'right', 'center'):
            raise ValueError("invalid padding scheme")
        self.padding_scheme = padding_scheme
        self.pad_dims = pad_dims
        self.ret_lengths = ret_lengths

    def pad_tensor(self, x, expected_size):
        assert(expected_size >= len(x))
        pad_amount = expected_size - len(x)
        if self.padding_scheme == 'left':
            padding = (pad_amount, 0)
        elif self.padding_scheme == 'right':
            padding = (0, pad_amount)
        else: # padding_scheme == 'center'
            padding = (pad_amount-pad_amount//2, pad_amount//2)
        x = torch.nn.functional.pad(x,
                                    padding,
                                    value=self.pad_value)
        return x

    def __call__(self, batch):
        single = False
        if not isinstance(batch[0], tuple):
            batch = [(b,) for b in batch]
            single = True
        batch = list(zip(*batch))

        for dim, X in enumerate(batch):
            if dim in self.pad_dims and torch.is_tensor(X[0]):
                lengths = torch.LongTensor([len(x) for x in X])
                max_length = lengths.max().item()
                sequences = torch.stack([self.pad_tensor(x,
                                                         expected_size=max_length)
                                         for x in X])
                if self.ret_lengths:
                    batch[dim] = sequences, lengths
                else:
                    batch[dim] = sequences
            else:
                batch[dim] = default_collate(list(X))
        if single:
            return batch[0]
        else:
            return tuple(batch)

class BoWCollate(object):
    """ Batch Collator intended to be used with EmbeddingBag.

    Example:
        >>> collate =  BoWCollate()
        >>> X_1, y_1 = torch.tensor([1, 2]), torch.tensor(1)
        >>> X_2, y_2 = torch.tensor([3]), torch.tensor(0)
        >>> collate([(X_1, y_1), (X_2, y_2)])
        (tensor([1, 2, 3]), torch([0, 2])), tensor([1., 0.])
    """
    def __call__(self, batch):
        target_elem = batch[0][1]
        if torch.is_tensor(target_elem):
            labels = torch.stack([x[1] for x in batch])
        elif isinstance(target_elem, np.ndarray):
            labels = torch.from_numpy(np.stack([x[1] for x in batch]))
        elif isinstance(target_elem, (float, int, list)):
            labels = torch.tensor([x[1] for x in batch])
        else:
            raise RuntimeError("Labels type not supported")
        texts = [entry[0] for entry in batch]
        offsets = [0] + [len(x) for x in texts]
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        texts = torch.cat(texts)
        return (texts, offsets), labels
