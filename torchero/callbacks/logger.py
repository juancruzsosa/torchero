from torchero.callbacks.base import Callback
from torchero.utils.format import format_metric


class Logger(Callback):
    UNRECOGNIZED_LEVEL = (
        "Unrecognized level {level}. Level parameter should be either 'epoch' "
        "or 'step'"
    )

    def __init__(self, separator=',\t', monitors=None, hparams=None, level='epoch'):
        self.separator = separator
        self.monitors = monitors
        self.hparams = hparams
        if level in ('epoch', 'step'):
            self.level = level
        else:
            raise ValueError(self.UNRECOGNIZED_LEVEL.format(level=repr(level)))

    def log(self):
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
        if self.level == 'step':
            self.log()

    def on_epoch_end(self):
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
