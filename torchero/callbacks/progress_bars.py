import logging
import sys

import tqdm

from torchero.callbacks.base import Callback
from torchero.utils.format import format_metric

from logging import StreamHandler


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, end='\n')
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

class ProgbarLogger(Callback):
    """ Callback that displays progress bars to monitor
        training/validation metrics
    """
    def __init__(self, ascii=False, notebook=False, monitors=None, hparams=None):
        """ Constructor

        Arguments:

            ascii (`bool`): if true display progress bar in ASCII mode.
            notebook (`bool`): Make outputs compatible for python notebooks
            monitor (list of str, optional): List of monitors to show after
                every status update. If not setted it will use all the monitors (default).
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
        self.trainer = trainer
        if self.notebook:
            self.trainer.logger.addHandler(TqdmLoggingHandler())
        else:
            self.trainer.logger.addHandler(TqdmLoggingHandler())

    def on_train_begin(self):
        """ Set up the training bars
        """
        self.epoch_tqdm = self.tqdm(total=self.trainer.total_epochs,
                                    unit='epoch',
                                    leave=True,
                                    position=0,
                                    ascii=self.ascii)
        self.epoch_bar = self.epoch_tqdm.__enter__()
        self.last_step = 0

    def on_epoch_begin(self):
        """ Add a new training bar
        """
        step_tqdm = self.tqdm(total=self.trainer.total_steps,
                              unit=' batchs',
                              leave=True,
                              position=1+len(self.step_tqdms),
                              ascii=self.ascii)
        self.step_tqdms.append(step_tqdm)
        self.step_bars.append(step_tqdm.__enter__())

    def on_log(self):
        """ Update the status of the progress bar
        with the level of epoch completion and the partial metrics
        """
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
        step_bar.set_description("Epoch {}".format(self.trainer.epoch+1))
        step_bar.set_postfix(**metrics, **hparams)
        step_bar.update(self.trainer.steps_trained - self.last_step)
        self.last_step = self.trainer.steps_trained

    def on_epoch_end(self):
        """ Close the current bar
        """
        self.epoch_bar.update()

    def on_train_end(self):
        """ Close all epoch bars
        """
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

    def __getstate__(self):
        return {
            "ascii": self.ascii,
            "notebook": self.notebook,
            "monitors": self.monitors,
            "hparams": self.hparams,
        }

    def __setstate__(self, state):
        self.ascii = state["ascii"]
        self.notebook = state["notebook"]
        if self.notebook:
            self.tqdm = tqdm.tqdm_notebook
        else:
            self.tqdm = tqdm.tqdm
        self.monitors = state["monitors"]
        self.hparams = state["hparams"]
        self.step_tqdms = []
        self.step_bars = []
