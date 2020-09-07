from torchero.callbacks.base import Callback
from torchero.utils.format import format_metric


class Logger(Callback):
    def __init__(self, separator=',\t', monitors=None, hparams=None):
        self.separator = separator
        self.monitors = monitors
        self.hparams = hparams

    def on_log(self):
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
        print("epoch: {trainer.epoch}/{trainer.total_epochs}{separator}"
              "step: {trainer.step}/{trainer.total_steps}{separator}"
              "{meters}".format(trainer=self.trainer,
                                meters=meters,
                                separator=self.separator))

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
