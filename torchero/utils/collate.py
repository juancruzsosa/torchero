import numpy as np
import torch


class PadSequenceCollate(object):
    def __init__(self, pad_value=0, pad_scheme='left'):
        self.pad_value = pad_value
        if pad_scheme not in ('left', 'right'):
            raise ValueError("invalid padding scheme")
        self.pad_scheme = pad_scheme

    def pad_tensor(self, x, expected_size):
        assert(expected_size >= len(x))
        padding = (expected_size - len(x), 0) if self.pad_scheme == 'left' else (0, expected_size - len(x))
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
