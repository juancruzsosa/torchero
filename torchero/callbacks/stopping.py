from torchero.callbacks.base import Callback
from torchero.utils.defaults import get_default_mode


class EarlyStopping(Callback):
    """ Callback for stop training when monitored metric not improve in time
    """
    UNRECOGNIZED_MODE_MESSAGE = (
        "Unrecognized mode {mode}. Options are: 'max', 'min', 'auto'"
    )
    INVALID_MODE_INFERENCE_MESSAGE = (
        "Could not infer mode from meter {meter}"
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
        criterion_by_mode = {'max': self._max_improved_criterion,
                             'min': self._min_improved_criterion}

        if self._mode.lower() == 'auto':
            self._mode = get_default_mode(self.trainer.meters[self.monitor])
            if self._mode == '':
                raise Exception(self.INVALID_MODE_INFERENCE_MESSAGE
                                    .format(meter=self.monitor))

        self._has_improved = criterion_by_mode[self._mode]

    def reset(self):
        self._last_best_monitor_value = None
        self._round = 0

    def _max_improved_criterion(self, value):
        return self._last_best_monitor_value - value < self._min_delta

    def _min_improved_criterion(self, value):
        return value - self._last_best_monitor_value < self._min_delta

    def on_log(self):
        monitor_value = self.trainer.metrics[self.monitor]

        if (self._last_best_monitor_value is not None and
                not self._has_improved(monitor_value)):
            self._round += 1
        else:
            self._round = 0

        if self._round > self._patience:
            self.trainer.stop_training()

        self._last_best_monitor_value = (self._last_best_monitor_value or
                                         monitor_value)

    @property
    def mode(self):
        return self._mode

    @property
    def min_delta(self):
        return self._min_delta

    @property
    def patience(self):
        return self._patience

    @property
    def monitor(self):
        return self._monitor
