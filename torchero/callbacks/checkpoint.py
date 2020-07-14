import os

import torch
import yaml

from torchero.callbacks.base import Callback
from torchero.callbacks.exceptions import MeterNotFound
from torchero.utils.defaults import get_default_mode


class ModelCheckpoint(Callback):
    """ Callback for checkpoint a model if it get betters in a given metric
    """
    UNRECOGNIZED_MODE = (
        "Unrecognized mode {mode}. Options are: 'max', 'min', 'auto'"
    )
    INVALID_MODE_INFERENCE_MESSAGE = (
        "Could not infer mode from meter {meter}"
    )

    def __init__(self, path, monitor, mode='auto'):
        """ Constructor

        Arguments:
            path (str): Path for the checkpoint file
            monitor (str): Metric name to monitor
            temp_dir (str): Temporary folder path.
            mode (str): One of 'max', 'min', 'auto'. Alters the checkpoint
            criterion to be based on maximum or minimum monitor quantity
            (respectively).
        """
        if mode not in ('max', 'min', 'auto'):
            raise Exception(self.UNRECOGNIZED_MODE.format(mode))

        self._mode = mode
        self.monitor_name = monitor
        self.path = path
        self.last_value = None
        self.outperform = False

    def criterion(self, mode):
        criterion_by_name = {'max': lambda b: self.last_value < b,
                             'min': lambda b: self.last_value > b}
        return criterion_by_name[mode.lower()]

    @property
    def mode(self):
        return self._mode

    def on_train_begin(self):
        if self._mode.lower() == 'auto':
            self._mode = get_default_mode(self.trainer.meters[self.monitor_name])
            if self._mode == '':
                raise Exception(self.INVALID_MODE_INFERENCE_MESSAGE
                                    .format(meter=self.monitor_name))

        self.is_better = self.criterion(self._mode)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if self.monitor_name not in self.trainer.meters_names():
            raise MeterNotFound(self.monitor_name)

    def load(self):
        """ Load checkpointed model
        """
        index_file = os.path.join(self.path, 'index.yaml')
        model_file = os.path.join(self.path, '0.pth')

        with open(index_file, 'r') as f:
            data = yaml.load(f)

        if self.monitor_name not in data[0]:
            raise MeterNotFound(self.monitor_name)

        self.last_value = data[0][self.monitor_name]
        self.trainer.model.load_state_dict(torch.load(model_file))

        return data[0]

    def on_epoch_end(self):
        if self.monitor_name not in self.trainer.metrics:
            raise MeterNotFound(self.monitor_name)

        value = self.trainer.metrics[self.monitor_name]
        if self.last_value is None or self.is_better(value):
            self.last_value = value
            index_content = [{self.monitor_name: self.last_value,
                              'epoch': self.trainer.epochs_trained}]

            index_file = os.path.join(self.path, 'index.yaml')
            model_file = os.path.join(self.path, '0.pth')

            with open(index_file, 'w') as index_file:
                yaml.dump(index_content, index_file)

            torch.save(self.trainer.model.state_dict(), model_file)
            self.outperform = True
