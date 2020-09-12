import sys

from torchero.callbacks.base import Callback
from torchero.utils.format import format_metric

from logging import StreamHandler

try:
    import tqdm
except ImportError:
    print("tqdm not installed!")
    print("install tqdm for progress bars support.")
    raise

class ProgbarLoggerStreamHandler(object):
    def __init__(self, tqdm=tqdm.tqdm, file=sys.stderr):
        self.file = file
        self.tqdm = tqdm

    def write(self, x):
        if len(x.rstrip()) > 0:
            self.tqdm.write(x, file=self.file, end='')

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()

class ProgbarLogger(Callback):
    """ Callback that displays progress bars to monitor
        training/validation metrics
    """

    def __init__(self, ascii=False, notebook=False, monitors=None, hparams=None):
        """ Constructor

        Arguments:

            ascii (`bool`): if true display progress bar in ASCII mode.
            notebook (`bool`): Make outputs compatible for python notebooks
            monitor (list of str): List of monitors to show after every status update
            hparams (list of str): List of hyperparameters names to show after every status update
        """
        self.ascii = ascii
        self.notebook = notebook
        if self.notebook:
            self.tqdm = tqdm.tqdm_notebook
        else:
            self.tqdm = tqdm.tqdm
        self.monitors = monitors
        self.hparams = hparams

        self.step_tqdms = []
        self.step_bars = []

    def accept(self, trainer):
        """ Accepts a trainer

        Args:
            trainer(instance of :class:`torchero.base.BaseTrainer`):
                Trainer to attach to
        """
        self.trainer = trainer
        if self.notebook:
            self.trainer.logger_handler.setStream(ProgbarLoggerStreamHandler(tqdm.notebook.tqdm))
        else:
            self.trainer.logger_handler.setStream(ProgbarLoggerStreamHandler(tqdm.tqdm))

    def on_train_begin(self):
        self.epoch_tqdm = self.tqdm(total=self.trainer.total_epochs,
                                    unit='epoch',
                                    leave=True,
                                    ascii=self.ascii)
        self.epoch_bar = self.epoch_tqdm.__enter__()
        self.last_step = 0

    def on_epoch_begin(self):
        step_tqdm = self.tqdm(total=self.trainer.total_steps,
                              unit=' batchs',
                              leave=True,
                              ascii=self.ascii)
        self.step_tqdms.append(step_tqdm)
        self.step_bars.append(step_tqdm.__enter__())

    def on_log(self):
        monitors = self.monitors
        if self.monitors is None:
            monitors = self.trainer.metrics.keys()


        hparams = self.hparams
        if self.hparams is None:
            hparams = self.trainer.hparams.keys()

        metrics = {name: format_metric(self.trainer.metrics[name])
                   for name in monitors
                   if name in self.trainer.metrics}
        hparams = {name: format_metric(self.trainer.hparams[name])
                   for name in hparams
                   if name in self.trainer.hparams}


        step_bar = self.step_bars[-1]
        step_bar.set_postfix(**metrics, **hparams)
        step_bar.update(self.trainer.steps_trained - self.last_step)
        self.last_step = self.trainer.steps_trained

    def on_epoch_end(self):
        self.epoch_bar.update()

    def on_train_end(self):
        self.epoch_bar.close()
        self.epoch_tqdm.close()

        for step_tqdm, step_bar in zip(self.step_tqdms, self.step_bars):
            step_bar.close()
            step_tqdm.close()

        self.step_tqdms = []
        self.step_bars = []

    def __repr__(self):
        monitors_repr = ""
        if self.monitors is not None:
            monitors_repr = ', monitors={}'.format(repr(self.monitors))
        hparams_repr = ""
        if self.hparams is not None:
            hparams_repr = ', hparams={}'.format(repr(self.hparams))
        return "{cls}(ascii={ascii}, notebook={notebook}{monitors}{hparams})".format(
            cls=self.__class__.__name__,
            monitors=monitors_repr,
            hparams=hparams_repr,
            ascii=repr(self.ascii),
            notebook=repr(self.notebook),
        )
