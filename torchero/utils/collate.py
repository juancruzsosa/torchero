import numpy as np
import torch


class PadSequenceCollate(object):
    def __init__(self, pad_value=0, padding_scheme='left'):
        self.pad_value = pad_value
        if padding_scheme not in ('left', 'right', 'center'):
            raise ValueError("invalid padding scheme")
        self.padding_scheme = padding_scheme

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
        # Sort the batch in the descending order
        max_length = max(map(lambda x: len(x[0]), batch))

        lengths = torch.LongTensor([len(x[0]) for x in batch])
        sequences = torch.stack([self.pad_tensor(x[0],
                                                 expected_size=max_length)
                                 for x in batch])
        target_elem = batch[0][1]
        if torch.is_tensor(target_elem):
            labels = torch.stack([x[1] for x in batch])
        elif isinstance(target_elem, np.ndarray):
            labels = torch.from_numpy(np.stack([x[1] for x in batch]))
        elif isinstance(target_elem, (float, int)):
            labels = torch.tensor([x[1] for x in batch])
        else:
            raise RuntimeError("Labels type not supported")
        return (sequences, lengths), labels

class BoWCollate(object):
    """ Batch Collator intended to be used with EmbeddingBag

    Example:
        >> collate = BoWCollate()
        >> collate(torch.tensor([([1, 2], torch.tensor(1.)),
                                 ([3], torch.tensor(0))])
        (torch.tensor([1, 2, 3]), torch.LongTensor([0, 2])), torch.tensor([1, 0])
    """
    def __call__(self, batch):
        target_elem = batch[0][1]
        if torch.is_tensor(target_elem):
            labels = torch.stack([x[1] for x in batch])
        elif isinstance(target_elem, np.ndarray):
            labels = torch.from_numpy(np.stack([x[1] for x in batch]))
        elif isinstance(target_elem, (float, int)):
            labels = torch.tensor([x[1] for x in batch])
        else:
            raise RuntimeError("Labels type not supported")
        texts = [entry[0] for entry in batch]
        offsets = [0] + [len(x) for x in texts]
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        texts = torch.cat(texts)
        return (texts, offsets), labels
