import tempfile
from abc import ABCMeta, abstractmethod

from torch.utils.data.dataset import Subset
import sys
import torch
import gc
import math
import logging

logger = logging.getLogger(__name__)

class BatchSizeFinder(object, metaclass=ABCMeta):
    """ Base class for finding optimal batch size for the given model
    """
    def __init__(self, model, min_size=1, max_size=None, shrink_factor=0.9, sort_dataset=True):
        """ Constructor

        Arguments:
            model (``torchero.models.Model``): Input model
            min_size (int): Minimum batch size to start from
            max_size (int, optional): Maximum batch size allowed. If None is passed
                the biggest possible batch size that don't raise OutOfMemory will be found
            shrink_factor (float): Ratio of batch_size to use instead the biggest one
                to prevent futures sporadics Out Of Memory in the future.
            sort_dataset (bool): Sort dataset first. Sorting is needed for
                datasets with variable size samples (like NLP datasets) to adjust
                the batch size to the worst case scenario
        """
        self.model = model
        if min_size < 0 or (max_size is not None and min_size > max_size):
            raise ValueError("Shrink size should be positive and lower than max_size")
        self.min_size = min_size
        self.max_size = max_size
        if shrink_factor > 1 or shrink_factor < 0:
            raise ValueError("Shrink factor should be between 0 and 1")
        self.shrink_factor = shrink_factor
        if self.max_size is not None and self.min_size > self.max_size:
            raise ValueError("invalid max_size argument: max_size must be greater than min_size")
        self.sort_dataset = sort_dataset

    @staticmethod
    def is_cuda_out_of_memory(exception):
        """ Returns True if it's a CUDA out of memory
        """
        return (
            len(exception.args) == 1 and "CUDA out of memory." in exception.args[0]
        )

    def test_batch_size(self, mode, batch_size, dataset, dl_params):
        dl_params['batch_size'] = batch_size
        dl = self.model._get_dataloader(dataset, **dl_params)
        batch = next(iter(dl))
        if mode == 'train':
            self.model.trainer._train_batch(batch)
        else: # eval/inference
            self.model.model.train(mode=False)
            with torch.no_grad():
                self.model.trainer.validator._process_batch(batch)
            self.model.model.train(mode=True)

    def free(self):
        gc.collect()
        torch.cuda.empty_cache()

    @abstractmethod
    def _find_batch_size(self, mode, dataset, dl_params):
        pass

    @staticmethod
    def _sort_dataset_by_size(dataset):
        logger.debug("Sorting dataset")
        sizes = torch.tensor([getsizeof(x) for x in dataset])
        indices = torch.argsort(sizes, descending=True).tolist()
        dataset = Subset(dataset, indices)
        return dataset

    def find(self, mode, dataset, **dl_params):
        """ Find the batch size for the given mode, and dataset

        Argument:
            mode (str): Either 'train' or 'inference'.
            dataset (``torch.utils.data.Dataset``): Dataset to use to find the learning rate
            **dl_params: Dataloader parameters
        """
        if self.sort_dataset:
            dataset = self._sort_dataset_by_size(dataset)
            dl_params['shuffle'] = False
        with tempfile.NamedTemporaryFile() as file:
            # Free cache
            self.free()
            # To not update weights by mistake save model to
            # a temporary file
            self.model.save(file.name)
            batch_size = self._find_batch_size(mode, dataset, dl_params)
            # Recover the model from the temporary file
            self.model.load(file.name)
            # Free cache
            self.free()
            return math.floor(batch_size * self.shrink_factor)

    def find_for_train(self, dataset, **dl_params):
        """ Find an adequate learning rate for training (forward + backward pass)

        Argument:
            dataset (``torch.utils.data.Dataset``): Dataset to use to find the learning rate
            **dl_params: Dataloader parameters
        """
        return self.find('train', dataset, **dl_params)

    def find_for_inference(self, dataset, **dl_params):
        """ Find an adequate learning rate for testing (only forward pass)

        Arguments:
            dataset (``torch.utils.data.Dataset``): Dataset to use to find the learning rate
            **dl_params: Dataloader parameters
        """
        return self.find('eval', dataset, **dl_params)

def getsizeof(x):
    if isinstance(x, (tuple, list)):
        return sum(getsizeof(y) for y in x)
    elif isinstance(x, dict):
        return sum(getsizeof(y) for y in x.values())
    elif torch.is_tensor(x):
        return x.nelement() * x.element_size()
    else:
        return sys.getsizeof(x)

class BatchSizeBinaryFinder(BatchSizeFinder):
    """ Module to find the optimal batch size using an binsearch strategy.

        First it grows the batch size by powers of two until it reaches Out of
        Memory, and then starts refining the final batch size by performing a
        binary search between the biggest batch size that worked and the lowest
        batch size that raised Out of Memory    First it grows the
    """
    def _find_batch_size(self, mode, dataset, dl_params):
        max_size = self.max_size or len(dataset)
        batch_size = self.min_size
        upper = None
        lower = None
        while upper is None or lower is None or upper - lower > 1:
            try:
                logger.debug("Testing with batch_size={}".format(batch_size))
                self.test_batch_size(mode, batch_size, dataset, dl_params)
            except RuntimeError as e:
                if self.is_cuda_out_of_memory(e):
                    upper = batch_size
                    if lower is None:
                        raise
                    logging.debug("Stepping down")
                    batch_size = math.floor((lower + upper)/2)
                    self.free()
                else:
                    raise
            else:
                lower = batch_size
                if batch_size >= max_size:
                    logging.debug("Max_size reached")
                    break
                if upper is None:
                    batch_size = min(max_size, batch_size * 2)
                else:
                    batch_size = math.ceil((lower + upper)/2)
        return lower

def AutoBatchSize(model, min_size=1, max_size=None, shrink_factor=0.95, sort_dataset=True):
    """ Finds the optimal batch size for training and inference

        Arguments:
            model (``torchero.models.Model``): Input model
            min_size (int): Minimum batch size to start from
            max_size (int, optional): Maximum batch size allowed. If None is passed
                the biggest possible batch size that don't raise OutOfMemory will be found
            shrink_factor (float): Ratio of batch_size to use instead the biggest one
                to prevent futures sporadics Out Of Memory in the future.
            sort_dataset (bool): Sort dataset first. Sorting is needed for
                datasets with variable size samples (like NLP datasets) to adjust
                the batch size to the worst case scenario
    """
    return BatchSizeBinaryFinder(model,
                                 min_size=1,
                                 max_size=max_size,
                                 shrink_factor=shrink_factor,
                                 sort_dataset=sort_dataset)
