from collections.abc import MutableMapping

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
                               ", ".join(map(lambda x: "{}={}".format(x[0], repr(x[1])), self)))
