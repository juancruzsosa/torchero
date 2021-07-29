import os

import torch
import yaml

from torchero.callbacks.base import Callback
from torchero.callbacks.exceptions import MeterNotFound
from torchero.utils.defaults import get_default_mode


class ModelCheckpoint(Callback):
    """ Callback to save the model if it improves in a given metric with
    respect to the previous epoch
    """
    UNRECOGNIZED_MODE = (
        "Unrecognized mode {mode}. Options are: 'max', 'min', 'auto'"
    )

    def __init__(self, path, monitor, mode='auto'):
        """ Constructor

        Arguments:
            path (str): Checkpoint path directory
            monitor (str): Metric name to monitor
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
        """ Returns the appropriate method to check if the model has improved
        """
        criterion_by_name = {'max': self._is_higher,
                             'min': self._is_lower}
        return criterion_by_name[mode.lower()]

    @property
    def mode(self):
        """ Checkpoint mode.
            'max' saves when the model maximizes a metric (accuracy, e.g f1_score)
            'min' saves when the model minimizes a metric (error, e.g rmse)
        """
        return self._mode

    def on_train_begin(self):
        """ Set-up directory structure
        """
        if self._mode.lower() == 'auto':
            self._mode = get_default_mode(self.trainer.meters[self.monitor_name])
        self.is_better = self.criterion(self._mode)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if self.monitor_name not in self.trainer.meters_names():
            raise MeterNotFound(self.monitor_name)

    def load(self):
        """ Load the checkpointed model
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
        """ Saves the model if it has improved
        """
        if self.monitor_name not in self.trainer.metrics:
            raise MeterNotFound(self.monitor_name)

        value = self.trainer.metrics[self.monitor_name]
        if self.last_value is None or self.is_better(value):
            if self.last_value is None:
                message = "Model saved to {path}"
            else:
                message = "Model saved to {path}: {monitor} improved from {last_value:.3f} to {current_value:.3f}"
            self.trainer.logger.info(message.format(path=repr(self.path),
                                                    monitor=self.monitor_name,
                                                    last_value=self.last_value,
                                                    current_value=value))
            self.last_value = value
            index_content = [{self.monitor_name: self.last_value,
                              'epoch': self.trainer.epochs_trained}]

            index_file = os.path.join(self.path, 'index.yaml')
            model_file = os.path.join(self.path, '0.pth')

            with open(index_file, 'w') as index_file:
                yaml.dump(index_content, index_file)

            torch.save(self.trainer.model.state_dict(), model_file)
            self.outperform = True

    def __repr__(self):
        return "{cls}(path={path}, monitor={monitor}, mode={mode})".format(
            cls=self.__class__.__name__,
            mode=repr(self._mode),
            monitor=repr(self.monitor_name),
            path=repr(self.path)
        )

    def _is_higher(self, new_val):
        return self.last_value < new_val

    def _is_lower(self, new_val):
        return self.last_value > new_val
