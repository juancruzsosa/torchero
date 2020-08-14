from torchero.callbacks.base import Callback
from torchero.utils.format import format_metric


class TensorBoardLogger(Callback):
    """ Callback to push metrics after epoch to Tensorboard
    """
    def __init__(self, writer, monitors=None):
        """ Constructor

        writer (SummaryWriter):
            Instance of `torch.utils.tensorboard.SummaryWriter
        monitors (list):
            List of metrics to log. If none is passed it will log all metrics
        """
        self.writer = writer
        self.monitors = monitors

    def on_epoch_end(self):
        monitors = self.monitors
        if monitors is None:
            monitors = self.trainer.metrics.keys()

        for name, value in self.trainer.train_metrics.items():
            self.writer.add_scalar('{}/Train'.format(name), value, global_step=self.trainer.epoch)

        for name, value in self.trainer.val_metrics.items():
            self.writer.add_scalar('{}/Test'.format(name), value, global_step=self.trainer.epoch)

        self.writer.flush()
