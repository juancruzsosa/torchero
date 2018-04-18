from .base import Hook

class History(Hook):
    """ Hook that record history of all training/validation metrics
    """
    def __init__(self):
        super(History, self).__init__()
        self.registry = []

    def log(self):
        line = {'epoch' : self.trainer.epoch,
                'step' : self.trainer.step}
        line.update(self.trainer.last_stats)
        self.registry.append(line)
