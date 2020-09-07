import torch
from collections.abc import Mapping, MutableMapping
from torchero.meters import ZeroMeasurementsError

class MetricsDict(Mapping):
    """ Container for hyperparameters dictionary
    """
    METER_ALREADY_EXISTS_MESSAGE = 'Meter {name} already exists as train meter'

    def __init__(self, mapping=(), **kwargs):
        """ Constructor

        Arguments:
            mapping (dict or list of key-val tuples):
                Build from existent mapping
        """
        self._meters = dict(mapping, **kwargs)
        self._metrics = {}

    def reset(self):
        self._metrics = {}

        for meter in self._meters.values():
            meter.reset()

    def compile(self):
        self._metrics = {}

        for metric_name, meter in self._meters.items():
            try:
                value = meter.value()
                self._metrics[metric_name] = value
            except ZeroMeasurementsError:
                continue

    def measure(self, output, y):
        with torch.no_grad():
            for meter in self._meters.values():
                meter.measure(output, y)

    def add_metric(self, name, meter):
        if name in self._meters:
            raise Exception(self.METER_ALREADY_EXISTS_MESSAGE
                                .format(name=name))

        self._meters[name] = meter

    @property
    def meters(self):
        return self._meters

    def __getitem__(self, key):
        """ Retrives the value of the given metric

        Arguments:
            key (str): Hyperparameter name

        Returns:
            The value of the metric ``key``
            if it exists, otherwise raises a KeyError
        """
        return self._metrics[key]

    def __delitem__(self, key):
        """ Removes a metric from the dict

        Arguments:
            key (str): Hyperparameter name to delete
        """
        del self._metrics[key]

    def __iter__(self):
        """ Same as keys() method
        """
        return iter(self._metrics)

    def keys(self):
        """ Returns an iterator for the hyperparametrics names
        """
        return self._metrics.keys()

    def values(self):
        """ Returns an iterator for the hyperparametrics values
        """
        return self._metrics.values()

    def items(self):
        """ Returns an iterator of (hyp. name, hyp. value) items
        """
        return self._metrics.items()

    def __len__(self):
        """ Returns the numbers of hyperparametrics
        """
        return len(self._metrics)

    def __repr__(self):
        return repr(self._metrics)

class ParamsDict(MutableMapping):
    """ Container for hyperparameters dictionary
    """
    def __init__(self, mapping=(), **kwargs):
        """ Constructor

        Arguments:
            mapping (dict or list of key-val tuples):
                Build from existent mapping
        """
        self._hparams = dict(mapping, **kwargs)

    def __getitem__(self, key):
        """ Retrives the value of the given hyperparameter

        Arguments:
            key (str): Hyperparameter name

        Returns:
            The value of the hyperparameter ``key``
            if it exists, otherwise raises a KeyError
        """
        return self._hparams[key].value

    def __delitem__(self, key):
        """ Removes a hyperparameter from the dict

        Arguments:
            key (str): Hyperparameter name to delete
        """
        del self._hparams[key]

    def __setitem__(self, key, value):
        """ Sets the value of a hyperparameter to a new value

        Arguments:
            key (str): Hyperparameter name
            value: New value for the hyperparameter
        """
        self._hparams[key].value = value

    def __iter__(self):
        """ Same as keys() method
        """
        return self.keys()

    def keys(self):
        """ Returns an iterator for the hyperparameters names
        """
        return iter(self._hparams.keys())

    def values(self):
        """ Returns an iterator for the hyperparameters values
        """
        yield from map(lambda x: x.value, self._hparams.values())

    def items(self):
        """ Returns an iterator of (hyp. name, hyp. value) items
        """
        yield from map(lambda x: (x[0], x[1].value), self._hparams.items())

    def __len__(self):
        """ Returns the numbers of hyperparameters
        """
        return len(self._hparams)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__,
                               ", ".join(map(lambda x: "{}={}".format(x[0], repr(x[1])), self.items())))
