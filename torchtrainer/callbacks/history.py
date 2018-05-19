from .base import Callback

class History(Callback):
    """ Callback that record history of all training/validation metrics
    """
    def __init__(self):
        super(History, self).__init__()
        self.registry = HistoryManager()

    def on_log(self):
        self.registry.append(self.trainer.epoch,
                            self.trainer.step,
                            self.trainer.metrics)

class HistoryManager(Callback):
    def __init__(self):
        self.records = []

    def __iter__(self):
        yield from self.records

    def __getitem__(self, idx):
        return self.records[idx]

    def __len__(self):
        return len(self.records)

    def append(self, epoch, step, metrics):
        self.records.append({'epoch' : epoch,
                             'step': step,
                             **metrics})
