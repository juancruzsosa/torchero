from torchero.callbacks.base import Callback
from torchero.utils.format import format_metric


class TensorBoardLogger(Callback):
    """ Callback to push metrics after every epoch to Tensorboard
    """
    def __init__(self, writer, monitors=None, test_writer=None):
        """ Constructor

        writer (SummaryWriter): Instance of `torch.utils.tensorboard.SummaryWriter`
        monitors (list): List of metrics to log. If none is passed it will log all metrics
        test_writer (SummaryWriter, optional): SummaryWriter for test metrics. If None is passed the same writer for both
        """
        self.writer = writer
        self.test_writer = test_writer
        self.monitors = monitors

    def on_epoch_end(self):
        """ Add the metrics for the new epoch to the board
        """
        monitors = self.monitors
        if monitors is None:
            monitors = self.trainer.metrics.keys()

        if self.test_writer is None:
            tag_fmt = '{metric}/{mode}'
            train_writer = self.writer
            test_writer = self.writer
        else:
            tag_fmt = '{metric}'
            train_writer = self.writer
            test_writer = self.test_writer

        for name, value in self.trainer.train_metrics.items():
            train_writer.add_scalar(tag_fmt.format(metric=name, mode='Train'),
                                    value, global_step=self.trainer.epochs_trained)

        for name, value in self.trainer.val_metrics.items():
            test_writer.add_scalar(tag_fmt.format(metric=name, mode='Test'),
                                   value, global_step=self.trainer.epochs_trained)

        self.writer.flush()
        self.test_writer.flush()
