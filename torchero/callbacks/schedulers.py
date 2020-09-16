from abc import abstractmethod

from torch.optim import lr_scheduler

from torchero.utils.defaults import get_default_mode
from torchero.callbacks.base import Callback


class OptimizerScheduler(Callback):
    """ Interface of Optimizer schedulers
    """
    def __init__(self, start=1, on_event='epoch_end', end=None, optimizer=None):
        if on_event not in ('log', 'epoch_end'):
            raise Exception("Unrecongized on_event parameter. Expected")
        self._on_event = on_event
        self._optimizer = optimizer
        if start < 1:
            raise ValueError("Start epoch should be an integer greater or equal than 1")
        if (end is not None) and (start > end):
            raise ValueError("End epoch should be an integer greater or equal than `start` argument")
        self._start = start
        self._end = end

    def accept(self, trainer):
        if self._optimizer is None and not hasattr(trainer, 'optimizer'):
            raise Exception("Trainer has not optimizer!")
        elif self._optimizer is None:
            self._optimizer = trainer.optimizer
        super(OptimizerScheduler, self).accept(trainer)

    def on_log(self):
        if self._on_event == 'log' and \
           self.trainer.epochs_trained >= self._start and \
           (self._end is None or self.trainer.epochs_trained <= self._end):
            self.step()

    def on_epoch_end(self):
        if self._on_event == 'epoch_end' and \
           self.trainer.epochs_trained >= self._start and \
           (self._end is None or self.trainer.epochs_trained <= self._end):
            self.step()

    @abstractmethod
    def step(self):
        pass


class _TorchScheduler(OptimizerScheduler):
    """ Adapts Torch learning rate scheduler to Torchero Optimzer scheduler
    """
    def __init__(self, params, start=0, end=None, on_event='epoch_end', optimizer=None, verbose=False):
        super(_TorchScheduler, self).__init__(start=start,
                                              end=end,
                                              on_event=on_event,
                                              optimizer=optimizer)
        self._params = params
        self.verbose = verbose

    def accept(self, trainer):
        super(_TorchScheduler, self).accept(trainer)
        self._scheduler = self._get_scheduler()

    def _step(self):
        self._scheduler.step()

    def _get_scheduler(self):
        return self.__class__.SCHEDULER_CLASS(optimizer=self._optimizer, **self._params)

    def step(self):
        if self.verbose:
            old_lrs = [param_group['lr'] for param_group in self._optimizer.param_groups]
        self._step()
        if self.verbose:
            for i, (param_group, old_lr) in enumerate(zip(self._optimizer.param_groups, old_lrs)):
                if param_group['lr'] < old_lr:
                    self.trainer.logger.info("Learning rate of param group n°{} reduced from {:.4e} to {:.4e}".format(i, old_lr, param_group['lr']))
                elif param_group['lr'] > old_lr:
                    self.trainer.logger.info("Learning rate of param group n°{} increased from {:.4e} to {:.4e}".format(i, old_lr, param_group['lr']))


class LambdaLR(_TorchScheduler):
    """ Adapter of torch.optim.lr_scheduler.LambdaLR
    """
    SCHEDULER_CLASS = lr_scheduler.LambdaLR

    def __init__(self, lr_lambda, start=1, end=None, on_event='epoch_end', optimizer=None, verbose=False):
        super(LambdaLR, self).__init__({'lr_lambda': lr_lambda},
                                       start=start,
                                       end=end,
                                       on_event=on_event,
                                       optimizer=optimizer,
                                       verbose=verbose)


class StepLR(_TorchScheduler):
    """ Adapter of torch.optim.lr_scheduler.StepLR
    """
    SCHEDULER_CLASS = lr_scheduler.StepLR

    def __init__(self,
                 step_size,
                 gamma=0.1,
                 start=1,
                 end=None,
                 on_event='epoch_end',
                 optimizer=None,
                 verbose=False):
        super(StepLR, self).__init__({'step_size': step_size,
                                      'gamma': gamma},
                                     start=start,
                                     end=end,
                                     on_event=on_event,
                                     optimizer=optimizer,
                                     verbose=verbose)


class MultiStepLR(_TorchScheduler):
    """ Adapter of torch.optim.lr_scheduler.MultiStepLR
    """
    SCHEDULER_CLASS = lr_scheduler.MultiStepLR

    def __init__(self,
                 milestones,
                 gamma=0.1,
                 start=0,
                 end=None,
                 on_event='epoch_end',
                 optimizer=None,
                 verbose=False):
        super(MultiStepLR, self).__init__({'milestones': milestones,
                                           'gamma': gamma},
                                          on_event=on_event,
                                          optimizer=optimizer,
                                          verbose=verbose)


class ExponentialLR(_TorchScheduler):
    """ Adapter of torch.optim.lr_scheduler.ExponentialLR
    """
    SCHEDULER_CLASS = lr_scheduler.ExponentialLR

    def __init__(self, gamma=0.1, on_event='epoch_end', optimizer=None, verbose=False):
        super(ExponentialLR, self).__init__({'gamma': gamma},
                                            on_event=on_event,
                                            optimizer=optimizer,
                                            verbose=verbose)


class CosineAnnealingLR(_TorchScheduler):
    """ Adapter of torch.optim.lr_scheduler.CosineAnnealingLR
    """
    SCHEDULER_CLASS = lr_scheduler.CosineAnnealingLR

    def __init__(self,
                 T_max,
                 eta_min=0,
                 start=1,
                 end=None,
                 on_event='epoch_end',
                 optimizer=None,
                 verbose=False):
        super(CosineAnnealingLR, self).__init__({'T_max': T_max,
                                                 'eta_min': eta_min},
                                                start=start,
                                                end=end,
                                                on_event=on_event,
                                                optimizer=optimizer,
                                                verbose=False)


class ReduceLROnPlateau(_TorchScheduler):
    """ Adapter of torch.optim.lr_scheduler.ReduceLROnPlateau
    """
    SCHEDULER_CLASS = lr_scheduler.ReduceLROnPlateau

    def __init__(self,
                 monitor,
                 mode='auto',
                 factor=0.1,
                 patience=3,
                 threshold=1e-4,
                 threshold_mode='rel',
                 cooldown=0,
                 min_lr=0,
                 eps=1e-8,
                 start=1,
                 end=None,
                 on_event='epoch_end',
                 optimizer=None,
                 verbose=True):
        super(ReduceLROnPlateau, self).__init__({
            'mode': mode,
            'factor': factor,
            'patience': patience,
            'threshold': threshold,
            'threshold_mode': threshold_mode,
            'cooldown': cooldown,
            'min_lr': min_lr,
            'eps': eps},
         start=start,
         end=end,
         on_event=on_event,
         optimizer=optimizer,
         verbose=verbose)
        self._mode = mode
        self._monitor = monitor

    def _get_scheduler(self):
        if self._params['mode'] == 'auto':
            self._params['mode'] = get_default_mode(self.trainer.meters[self._monitor])
        return super(ReduceLROnPlateau, self)._get_scheduler()

    def _step(self):
        if self._monitor in self.trainer.metrics:
            self._scheduler.step(self.trainer.metrics[self._monitor])
