from abc import abstractmethod

from torch.optim import lr_scheduler

from torchero.callbacks.base import Callback
from torchero.callbacks.monitor import LRMonitor


class OptimizerScheduler(Callback):
    """ Interface of Optimizer schedulers
    """
    def __init__(self, on_event='epoch_begin', optimizer=None):
        if on_event not in ('log', 'epoch_begin', 'epoch_end'):
            raise Exception("Unrecongized on_event parameter. Expected")
        self._on_event = on_event
        self._optimizer = optimizer

    def accept(self, trainer):
        if self._optimizer is None and not hasattr(trainer, 'optimizer'):
            raise Exception("Trainer has not optimizer!")
        elif self._optimizer is None:
            self._optimizer = trainer.optimizer
        super(OptimizerScheduler, self).accept(trainer)
        trainer.add_callback(LRMonitor(optimizer=self._optimizer))

    def on_log(self):
        if self._on_event == 'log':
            self.step()

    def on_epoch_begin(self):
        if self._on_event == 'epoch_begin':
            self.step()

    def on_epoch_end(self):
        if self._on_event == 'epoch_end':
            self.step()

    @abstractmethod
    def step(self):
        pass


class _TorchScheduler(OptimizerScheduler):
    """ Adapts Torch learning rate scheduler to Torchero Optimzer scheduler
    """
    def __init__(self, params, on_event='epoch_begin', optimizer=None):
        super(_TorchScheduler, self).__init__(on_event=on_event,
                                              optimizer=optimizer)
        self._params = params

    def accept(self, trainer):
        super(_TorchScheduler, self).accept(trainer)
        self._scheduler = self.__class__.SCHEDULER_CLASS(
                                optimizer=self._optimizer,
                                **self._params)

    def step(self):
        self._scheduler.step()


class LambdaLR(_TorchScheduler):
    """ Adapter of torch.optim.lr_scheduler.LambdaLR
    """
    SCHEDULER_CLASS = lr_scheduler.LambdaLR

    def __init__(self, lr_lambda, on_event='epoch_begin', optimizer=None):
        super(LambdaLR, self).__init__({'lr_lambda': lr_lambda},
                                       on_event=on_event,
                                       optimizer=optimizer)


class StepLR(_TorchScheduler):
    """ Adapter of torch.optim.lr_scheduler.StepLR
    """
    SCHEDULER_CLASS = lr_scheduler.StepLR

    def __init__(self,
                 step_size,
                 gamma=0.1,
                 on_event='epoch_begin',
                 optimizer=None):
        super(StepLR, self).__init__({'step_size': step_size, 'gamma': gamma},
                                     on_event=on_event,
                                     optimizer=optimizer)


class MultiStepLR(_TorchScheduler):
    """ Adapter of torch.optim.lr_scheduler.MultiStepLR
    """
    SCHEDULER_CLASS = lr_scheduler.MultiStepLR

    def __init__(self,
                 milestones,
                 gamma=0.1,
                 on_event='epoch_begin',
                 optimizer=None):
        super(MultiStepLR, self).__init__({'milestones': milestones,
                                           'gamma': gamma},
                                          on_event=on_event,
                                          optimizer=optimizer)


class ExponentialLR(_TorchScheduler):
    """ Adapter of torch.optim.lr_scheduler.ExponentialLR
    """
    SCHEDULER_CLASS = lr_scheduler.ExponentialLR

    def __init__(self, gamma=0.1, on_event='epoch_begin', optimizer=None):
        super(ExponentialLR, self).__init__({'gamma': gamma},
                                            on_event=on_event,
                                            optimizer=optimizer)


class CosineAnnealingLR(_TorchScheduler):
    """ Adapter of torch.optim.lr_scheduler.CosineAnnealingLR
    """
    SCHEDULER_CLASS = lr_scheduler.CosineAnnealingLR

    def __init__(self,
                 T_max,
                 eta_min=0,
                 on_event='epoch_begin',
                 optimizer=None):
        super(CosineAnnealingLR, self).__init__({'T_max': T_max,
                                                 'eta_min': eta_min},
                                                on_event=on_event,
                                                optimizer=optimizer)


class ReduceLROnPlateau(_TorchScheduler):
    """ Adapter of torch.optim.lr_scheduler.ReduceLROnPlateau
    """
    SCHEDULER_CLASS = lr_scheduler.ReduceLROnPlateau

    def __init__(self,
                 monitor,
                 mode='min',
                 factor=0.1,
                 patience=3,
                 threshold=1e-4,
                 threshold_mode='rel',
                 cooldown=0,
                 min_lr=0,
                 eps=1e-8,
                 on_event='epoch_end',
                 optimizer=None):
        super(ReduceLROnPlateau, self).__init__({
            'mode': mode,
            'factor': factor,
            'patience': patience,
            'threshold': threshold,
            'threshold_mode': threshold_mode,
            'cooldown': cooldown,
            'min_lr': min_lr,
            'eps': eps},
         on_event=on_event,
         optimizer=optimizer)
        self._monitor = monitor

    def step(self):
        if self._monitor in self.trainer.metrics:
            self._scheduler.step(self.trainer.metrics[self._monitor])
