from abc import abstractmethod, ABCMeta

from torch.optim import lr_scheduler

from torchero.utils.defaults import get_default_mode
from torchero.callbacks.base import Callback


class OptimizerScheduler(Callback, metaclass=ABCMeta):
    """ Abstract class for all optimizer schedulers
    """
    def __init__(self, start=1, on_event='epoch_end', end=None, optimizer=None):
        """ Constructor

        Arguments:
            on_event (str): Either 'epoch_end' or 'log'. Defines when the
                scheduler takes place. 'epoch_end' to schedule after each epoch,
                'log' to scheduler after every log. Default: 'epoch_end'
            start (`int`): Epoch when the scheduling starts applying. Default: 1
            end (`int`): Epoch when the scheduling ends applying. Default: None.
            optimizer (`torch.optim.Optimizer`):
                Wrapped optimizer, if optimizer is None is passed it uses the
                trainer optimizer. Default: None
        """
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
        """ Adjusts the learning rate at the end of the epoch if its configured per step
        """
        if self._on_event == 'log' and \
           self.trainer.epochs_trained >= self._start and \
           (self._end is None or self.trainer.epochs_trained <= self._end):
            self.step()

    def on_epoch_end(self):
        """ Adjusts the learning rate at the end of the epoch if its configured per epoch
        """
        if self._on_event == 'epoch_end' and \
           self.trainer.epochs_trained >= self._start and \
           (self._end is None or self.trainer.epochs_trained <= self._end):
            self.step()

    @abstractmethod
    def step(self):
        """ Adjust the learning rate
        """
        pass


class _TorchScheduler(OptimizerScheduler):
    """ Callback Wrapper for torch.optim.lr_scheduler modules
    """
    def __init__(self, params, start=1, end=None, on_event='epoch_end', optimizer=None, verbose=False):
        """ Constructor

        Arguments:
            params (dict): Parameters
            start (`int`): Epoch when the scheduling starts applying. Default: 1
            end (`int`): Epoch when the scheduling ends applying. Default: None.
            on_event (str): Either 'epoch_end' or 'log'. Defines when the
                scheduler takes place. 'epoch_end' to schedule after each epoch,
                'log' to scheduler after every log. Default: 'epoch_end'
            optimizer (`torch.optim.Optimizer`):
                Wrapped optimizer, if optimizer is None is passed it uses the
                trainer optimizer. Default: None
        """
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
        """ Returns the respective torch scheduler
        """
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
    """ Sets the learning rate of each parameter group to the initial lr times
    a given function.
    """
    SCHEDULER_CLASS = lr_scheduler.LambdaLR

    def __init__(self, lr_lambda, start=1, end=None, on_event='epoch_end', optimizer=None, verbose=False):
        """ Constructor

        Arguments:
            lr_lambda (function or list): A function which computes a
                multiplicative factor given an integer parameter epoch, or a list
                of such functions, one for each group in optimizer.param_groups.
            start (`int`): Epoch when the scheduling starts applying. Default: 1
            end (`int`): Epoch when the scheduling ends applying. Default: None.
            on_event (str): Either 'epoch_end' or 'log'. Defines when the
                scheduler takes place. 'epoch_end' to schedule after each epoch,
                'log' to scheduler after every log. Default: 'epoch_end'
            optimizer (`torch.optim.Optimizer`): Wrapped optimizer, if
                optimizer is None is passed it uses the trainer optimizer. Default:
                None
            verbose (bool): If True logs a training message after each update in the
                learning rate.
        """
        super(LambdaLR, self).__init__({'lr_lambda': lr_lambda},
                                       start=start,
                                       end=end,
                                       on_event=on_event,
                                       optimizer=optimizer,
                                       verbose=verbose)


class StepLR(_TorchScheduler):
    """ Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler.
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
        """ Constructor

        Arguments:
            step_size (int): Period of learning rate decay.
            gamma (float): Multiplicative factor of learning rate decay.
                Default: 0.1.
            start (`int`): Epoch when the scheduling starts applying. Default: 1
            end (`int`): Epoch when the scheduling ends applying. Default: None.
            on_event (str): Either 'epoch_end' or 'log'. Defines when the
                scheduler takes place. 'epoch_end' to schedule after each epoch,
                'log' to scheduler after every log. Default: 'epoch_end'
            optimizer (`torch.optim.Optimizer`): Wrapped optimizer, if
                optimizer is None is passed it uses the trainer optimizer. Default:
                None
            verbose (bool): If True logs a training message after each update in the
                learning rate.
        """
        super(StepLR, self).__init__({'step_size': step_size,
                                      'gamma': gamma},
                                     start=start,
                                     end=end,
                                     on_event=on_event,
                                     optimizer=optimizer,
                                     verbose=verbose)


class MultiStepLR(_TorchScheduler):
    """ Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler.
    """
    SCHEDULER_CLASS = lr_scheduler.MultiStepLR

    def __init__(self,
                 milestones,
                 gamma=0.1,
                 start=1,
                 end=None,
                 on_event='epoch_end',
                 optimizer=None,
                 verbose=False):
        """ Constructor

        Arguments:
            milestones (list): List of epoch indices. Must be increasing.
            gamma (float): Multiplicative factor of learning rate decay.
                Default: 0.1.
            start (`int`): Epoch when the scheduling starts applying. Default: 1
            end (`int`): Epoch when the scheduling ends applying. Default: None.
            on_event (str): Either 'epoch_end' or 'log'. Defines when the
                scheduler takes place. 'epoch_end' to schedule after each epoch,
                'log' to scheduler after every log. Default: 'epoch_end'
            optimizer (`torch.optim.Optimizer`): Wrapped optimizer, if
                optimizer is None is passed it uses the trainer optimizer. Default:
                None
            verbose (bool): If True logs a training message after each update in the
                learning rate.
        """
        super(MultiStepLR, self).__init__({'milestones': milestones,
                                           'gamma': gamma},
                                          start=start,
                                          end=end,
                                          on_event=on_event,
                                          optimizer=optimizer,
                                          verbose=verbose)


class ExponentialLR(_TorchScheduler):
    """ Decays the learning rate of each parameter group by gamma every epoch.
    """
    SCHEDULER_CLASS = lr_scheduler.ExponentialLR

    def __init__(self, gamma=0.1, start=1, end=None, on_event='epoch_end', optimizer=None, verbose=False):
        super(ExponentialLR, self).__init__({'gamma': gamma},
                                            start=start,
                                            end=end,
                                            on_event=on_event,
                                            optimizer=optimizer,
                                            verbose=verbose)


class CosineAnnealingLR(_TorchScheduler):
    """ Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR.

    See documentation of ``torch.optim.lr_scheduler.CosineAnnealingLR`` for more info.
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
        """ Constructor

        Arguments:
            T_max (int): Maximum number of iterations.
            eta_min (float): Minimum learning rate. Default: 0.
            start (`int`): Epoch when the scheduling starts applying. Default: 1
            end (`int`): Epoch when the scheduling ends applying. Default: None.
            on_event (str): Either 'epoch_end' or 'log'. Defines when the
                scheduler takes place. 'epoch_end' to schedule after each epoch,
                'log' to scheduler after every log. Default: 'epoch_end'
            optimizer (`torch.optim.Optimizer`): Wrapped optimizer, if
                optimizer is None is passed it uses the trainer optimizer. Default:
                None
            verbose (bool): If True logs a training message after each update in the
                learning rate.
        """
        super(CosineAnnealingLR, self).__init__({'T_max': T_max,
                                                 'eta_min': eta_min},
                                                start=start,
                                                end=end,
                                                on_event=on_event,
                                                optimizer=optimizer,
                                                verbose=False)


class ReduceLROnPlateau(_TorchScheduler):
    """ Reduce learning rate when a metric has stopped improving. Models often
    benefit from reducing the learning rate by a factor of 2-10 once learning
    stagnates. This scheduler reads a metrics quantity and if no improvement is
    seen for a 'patience' number of epochs, the learning rate is reduced.
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
        """ Constructor

        Arguments:
            monitor (str): Number of the metric to track to adjust learning rate.
            mode (str): One of `min`, `max`, `auto`. In `auto` the mode will be
                captured from the monitor at runtime; `min` mode, lr will be
                reduced when the quantity monitored has stopped decreasing; in
                `max` mode it will be reduced when the quantity monitored has
                stopped increasing. Default: 'auto'.
            factor (float): Factor by which the learning rate will be
                reduced. new_lr = lr * factor. Default: 0.1.
            patience (int): Number of epochs with no improvement after
                which learning rate will be reduced. For example, if
                `patience = 2`, then we will ignore the first 2 epochs
                with no improvement, and will only decrease the LR after the
                3rd epoch if the loss still hasn't improved then.
                Default: 10.
            threshold (float): Threshold for measuring the new optimum,
                to only focus on significant changes. Default: 1e-4.
            threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
                dynamic_threshold = best * ( 1 + threshold ) in 'max'
                mode or best * ( 1 - threshold ) in `min` mode.
                In `abs` mode, dynamic_threshold = best + threshold in
                `max` mode or best - threshold in `min` mode. Default: 'rel'.
            cooldown (int): Number of epochs to wait before resuming
                normal operation after lr has been reduced. Default: 0.
            min_lr (float or list): A scalar or a list of scalars. A
                lower bound on the learning rate of all param groups
                or each group respectively. Default: 0.
            eps (float): Minimal decay applied to lr. If the difference
                between new and old lr is smaller than eps, the update is
                ignored. Default: 1e-8.
            start (`int`): Epoch when the scheduling starts applying. Default: 1
            end (`int`): Epoch when the scheduling ends applying. Default: None.
            on_event (str): Either 'epoch_end' or 'log'. Defines when the
                scheduler takes place. 'epoch_end' to schedule after each epoch,
                'log' to scheduler after every log. Default: 'epoch_end'
            optimizer (`torch.optim.Optimizer`): Wrapped optimizer, if
                optimizer is None is passed it uses the trainer optimizer. Default: None
            verbose (bool): If True logs a training message after each update
                in the learning rate.
        """
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
