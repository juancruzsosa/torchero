from .base import Callback

class History(Callback):
    """ Callback that record history of all training/validation metrics
    """
    def __init__(self):
        super(History, self).__init__()
        self.registry = []

    def on_log(self):
        line = {'epoch' : self.trainer.epoch,
                'step' : self.trainer.step}
        line.update(self.trainer.metrics)
        self.registry.append(line)
