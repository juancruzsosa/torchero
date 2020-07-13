from torchero.callbacks.base import Callback
from torchero.utils.format import format_metric


class Logger(Callback):
    def __init__(self, separator=',\t', monitors=None):
        self.separator = separator
        self.monitors = monitors

    def on_log(self):
        monitors = self.monitors
        if monitors is None:
            monitors = self.trainer.metrics.keys()

        metrics = {name: format_metric(self.trainer.metrics[name])
                   for name in monitors
                   if name in self.trainer.metrics}

        meters = self.separator.join(map(lambda x: '{}: {}'.format(*x),
                                         metrics.items()))
        print("epoch: {trainer.epoch}/{trainer.total_epochs}{separator}"
              "step: {trainer.step}/{trainer.total_steps}{separator}"
              "{meters}".format(trainer=self.trainer,
                                meters=meters,
                                separator=self.separator))
