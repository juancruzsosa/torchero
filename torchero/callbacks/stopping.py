from torchero.callbacks.base import Callback
from torchero.utils.defaults import get_default_mode


class EarlyStopping(Callback):
    """ Callback to stop training when metric has stopped improving
    """
    UNRECOGNIZED_MODE_MESSAGE = (
        "Unrecognized mode {mode}. Options are: 'max', 'min', 'auto'"
    )

    def __init__(self, monitor, min_delta=0, patience=0, mode='auto'):
        """ Constructor

        Arguments:
            monitor(str): Metric name to monitor
            min_delta(float): Minimum margin if not improvement
            patience(int): Number of steps of not improvement after stop
            training
            mode(str): One of 'max', 'min', 'auto'. Alters the improvement
            criterion to be based on maximum or minimum monitor quantity
        """
        super(EarlyStopping, self).__init__()

        if mode not in ('max', 'min', 'auto'):
            raise ValueError(self.UNRECOGNIZED_MODE_MESSAGE.format(mode=mode))

        self._monitor = monitor
        self._mode = mode
        self._patience = patience
        self._min_delta = min_delta
        self.reset()

    def on_train_begin(self):
        """ Selects the appropriate improvement criterion
        """
        criterion_by_mode = {'max': self._max_improved_criterion,
                             'min': self._min_improved_criterion}

        if self._mode.lower() == 'auto':
            self._mode = get_default_mode(self.trainer.meters[self.monitor])

        self._has_improved = criterion_by_mode[self._mode]

    def reset(self):
        self._last_best_monitor_value = None
        self._round = 0

    def _max_improved_criterion(self, value):
        return self._last_best_monitor_value + self._min_delta < value

    def _min_improved_criterion(self, value):
        return self._last_best_monitor_value - self._min_delta > value

    def step(self):
        """ Stops the training if the model has not improved
        """
        if self.monitor not in self.trainer.metrics:
            return
        monitor_value = self.trainer.metrics[self.monitor]

        if (self._last_best_monitor_value is not None and
                not self._has_improved(monitor_value)):
            self._round += 1
        else:
            self._round = 0
            self._last_best_monitor_value = monitor_value

        if self._round > self._patience:
            self.trainer.logger.info("{monitor} has not improved from {value:.4f} for {round} rounds: Stop Training...".format(monitor=self._monitor,
                                                                                                                               value=self._last_best_monitor_value,
                                                                                                                               round=self._round))
            self.trainer.stop_training()

    def on_epoch_end(self):
        self.step()

    @property
    def mode(self):
        """ Early stopping mode.
            'max' stops training when the model not maximizes a metric (accuracy, e.g f1_score)
            'min' stops training when the model not minimizes a metric (error, e.g rmse)
        """
        return self._mode

    @property
    def min_delta(self):
        """ Minimum margin if not improvement
        """
        return self._min_delta

    @property
    def patience(self):
        """ Number of steps of not improvement after stop
            training
        """
        return self._patience

    @property
    def monitor(self):
        """ Metric to pay attention to
        """
        return self._monitor

    def __repr__(self):
        return "{cls}(monitor={monitor}, min_delta={min_delta}, patience={patience}, mode={mode})".format(
            cls=self.__class__.__name__,
            monitor=repr(self._monitor),
            mode=repr(self._mode),
            patience=repr(self._patience),
            min_delta=repr(self._min_delta)
        )
