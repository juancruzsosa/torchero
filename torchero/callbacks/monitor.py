from torchero.callbacks.base import Callback


class Monitor(Callback):
    def __init__(self, property_name, property_fn):
        super(Monitor, self).__init__()
        self._property_name = property_name
        self._property_fn = property_fn

    def on_log(self):
        self.trainer._train_metrics[self._property_name] = self._property_fn()


class OptimizerMonitor(Monitor):
    def __init__(self, property_name, property_fn, optimizer=None):
        super(OptimizerMonitor, self).__init__(property_name, property_fn)
        self._optimizer = optimizer

    def accept(self, trainer):
        if self._optimizer is None and not hasattr(trainer, 'optimizer'):
            raise Exception("Trainer has not optimizer!")
        elif self._optimizer is None:
            self._optimizer = trainer.optimizer
        super(OptimizerMonitor, self).accept(trainer)


class LRMonitor(OptimizerMonitor):
    def __init__(self, optimizer=None):
        super(LRMonitor, self).__init__(property_name='lr',
                                        property_fn=self.property,
                                        optimizer=optimizer)

    def property(self):
        res = [pg['lr'] for pg in self._optimizer.param_groups]
        if len(res) == 1:
            return res[0]
        else:
            return res
