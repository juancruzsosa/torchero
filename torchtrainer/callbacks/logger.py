from .base import Callback

class Logger(Callback):
    def __init__(self, separator=',\t'):
        self.separator = separator

    def format(self, value):
        if isinstance(value, float):
            return '{:.3f}'.format(value)
        else:
            return str(value)

    def on_log(self):
        metrics = {name: self.format(value)
                   for name, value in self.trainer.metrics.items()}
        meters = self.separator.join(map(lambda x: '{}: {}'.format(*x), metrics.items()))
        print("epoch: {trainer.epoch}/{trainer.total_epochs}{separator}"
              "step: {trainer.step}/{trainer.total_steps}{separator}"
              "{meters}".format(trainer=self.trainer,
                                meters=meters,
                                separator=self.separator))
