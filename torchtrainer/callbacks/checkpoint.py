import os
import tempfile
import shutil
import yaml
import torch
from .base import Callback
from .exceptions import MeterNotFound

class ModelCheckpoint(Callback):
    """ Callback for checkpoint a model if it get betters in a given metric
    """
    UNRECOGNIZED_MODE = "Unrecognized mode {mode}. Options are: 'max', 'min'"

    def __init__(self, path, monitor, mode='min', temp_dir=None):
        """ Constructor

        Arguments:
            path (str): Path for the checkpoint file
            monitor (str): Metric name to monitor
            temp_dir (str): Temporary folder path.
            mode (str): One of 'max' or 'min'. Alters the checkpoint criterion
            to be based on maximum or minimum monitor quantity (respectively).
        """

        self.monitor_name = monitor
        self.path = path
        self.last_value = None
        self.temp_dirname = temp_dir
        self.outperform = False

        if mode.lower() == 'min':
            self.is_better = lambda value: self.last_value > value
        elif mode.lower() == 'max':
            self.is_better = lambda value: self.last_value < value
        else:
            raise Exception(self.UNRECOGNIZED_MODE.format(mode))

    def on_train_begin(self):
        if self.monitor_name not in self.trainer.meters_names():
            raise MeterNotFound(self.monitor_name)
        self.temp_dir = tempfile.mkdtemp(dir=self.temp_dirname)

    def load(self):
        """ Load checkpointed model
        """
        try:
            extract_dir = tempfile.mkdtemp(dir=self.temp_dirname)
            shutil.unpack_archive(self.path + '.zip', extract_dir)

            with open(os.path.join(extract_dir, 'index.yaml'), 'r') as f:
                data = yaml.load(f)

            if self.monitor_name not in data[0]:
                raise MeterNotFound(self.monitor_name)

            self.last_value = data[0][self.monitor_name]
            self.trainer.model.load_state_dict(torch.load(os.path.join(extract_dir, '0.pth')))
        finally:
            shutil.rmtree(extract_dir)

        return data[0]

    def on_epoch_end(self):
        if self.monitor_name not in self.trainer.metrics:
            shutil.rmtree(self.temp_dir)
            raise MeterNotFound(self.monitor_name)

        value = self.trainer.metrics[self.monitor_name]
        if self.last_value is None or self.is_better(value):
            self.last_value = value
            index_content = [{self.monitor_name: self.last_value,
                              'epoch': self.trainer.epochs_trained}]

            with open(os.path.join(self.temp_dir, 'index.yaml'), 'w') as index_file:
                yaml.dump(index_content, index_file)

            torch.save(self.trainer.model.state_dict(), os.path.join(self.temp_dir, '0.pth'))
            self.outperform = True

    def on_train_end(self):
        if self.outperform:
            shutil.make_archive(self.path, 'zip', self.temp_dir)
        shutil.rmtree(self.temp_dir)
        del self.temp_dir
