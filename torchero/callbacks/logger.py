from torchero.callbacks.base import Callback
from torchero.utils.format import format_metric


class Logger(Callback):
    """ Callback to log metrics to a stdout after each epoch
    """
    UNRECOGNIZED_LEVEL = (
        "Unrecognized level {level}. Level parameter should be either 'epoch' "
        "or 'step'"
    )

    def __init__(self, separator=',\t', monitors=None, hparams=None, level='epoch'):
        """ Constructor

        Arguments:
            separator (str): String to separate columns
            monitors (list of str): Set of metrics names to report. If None
                (default) is passed it will show all metrics.
            hparams (list of str): Set of hyperparameters to report.
            level (str): When to log the metrics.
                level='epoch' means after every epoch (default)
                level='step' after every iteration step
        """
        self.separator = separator
        self.monitors = monitors
        self.hparams = hparams
        if level in ('epoch', 'step'):
            self.level = level
        else:
            raise ValueError(self.UNRECOGNIZED_LEVEL.format(level=repr(level)))

    def log(self):
        """ Add a new line in the file
        """
        monitors = self.monitors
        if monitors is None:
            monitors = self.trainer.metrics.keys()

        hparams = self.hparams
        if hparams is None:
            hparams = self.trainer.hparams.keys()

        metrics = {name: format_metric(self.trainer.metrics[name])
                   for name in monitors
                   if name in self.trainer.metrics}

        metrics.update({name: format_metric(self.trainer.hparams[name])
                        for name in hparams
                        if name in self.trainer.hparams})

        meters = self.separator.join(map(lambda x: '{}: {}'.format(*x),
                                         metrics.items()))

        message = "epoch: {trainer.epoch}/{trainer.total_epochs}{separator}"
        if self.level == 'step':
            message += "step: {trainer.step}/{trainer.total_steps}{separator}"
        message += "{meters}"
        self.trainer.logger.info(message.format(trainer=self.trainer,
                                                meters=meters,
                                                separator=self.separator))

    def on_log(self):
        """ Add a new line in the file if its configured per step
        """
        if self.level == 'step':
            self.log()

    def on_epoch_end(self):
        """ Add a new line in the file if its configured per epoch
        """
        if self.level == 'epoch':
            self.log()

    def __repr__(self):
        monitors_repr = ""
        if self.monitors is not None:
            monitors_repr = 'monitors={}'.format(repr(self.monitors))
        hparams_repr = ""
        if self.hparams is not None:
            hparams_repr = 'hparams={}'.format(repr(self.hparams))
        return "{cls}(separator={sep}{monitors}{hparams})".format(
            cls=self.__class__.__name__,
            monitors=monitors_repr,
            hparams=hparams_repr
        )
