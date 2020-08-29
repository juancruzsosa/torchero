from abc import abstractmethod, ABCMeta
from itertools import islice

class P(object, metaclass=ABCMeta):
    """ Base class for all HyperParameters
    """
    def __init__(self):
        """ Constructor
        """
        self.trainer = None

    def accept(self, trainer):
        """ Accept a trainer

        Args:
            trainer(instance of :class:`torchero.base.BaseTrainer`):
                Trainer to attach to
        """
        self.trainer = trainer

    @property
    @abstractmethod
    def value(self):
        pass

class FixedP(P):
    """ Hyperparameter for static values
    (values not dependent of any other component)
    """
    def __init__(self, value):
        """ Constructor

        Arguments:
            value: Value for this hyperparameter
        """
        super(FixedP, self).__init__()
        self._value = value

    @property
    def value(self):
        """ Current value of the hyperparameter
        """
        return self._value

    @value.setter
    def value(self, new_value):
        """ Sets another value to the given hyperparameter
        """
        self._value = new_value

class LambdaP(P):
    """ Hyperparameter for functions of trainer
    """
    def __init__(self, f):
        """ Constructor

        Arguments:
            f (callable): Function of a trainer
        """
        super(LambdaP, self).__init__()
        self.f = f

    @property
    def value(self):
        """ Current value of the hyperparameter
        """
        if self.trainer is None:
            raise Exception("Trainer not attached!")
        return self.f(self.trainer)

    @value.setter
    def value(self, new_f):
        """ Sets another value to the given hyperparameter
        """
        self.f = new_f

class OptimP(P):
    def __init__(self, property_name):
        super(OptimP, self).__init__()
        p_list = property_name.split('.')
        self.property_name = p_list[0]
        self.optimizer_name = None

        self.property_arg = None
        if len(p_list) > 1:
            self.property_arg = p_list[1]

    @staticmethod
    def default_optimizer_name(optimizer):
        return optimizer.__class__.__name__

    @staticmethod
    def _param_group(optimizer, i):
        param_groups = optimizer.param_groups
        try:
            param_groups = islice(param_groups, i, None)
            return next(iter(param_groups))
        except StopIteration:
            raise IndexError("{} optimizer has not #{} param group".format(OptimP.default_optimizer_name(optimizer), i))

    def accept(self, trainer):
        if self.property_name not in ['name']:
            param_nr = int(self.property_arg) if self.property_arg is not None else 0
            param_group = self._param_group(trainer.optimizer, param_nr)
            if self.property_name not in param_group:
                raise Exception("{} optimizer has not property '{}'!".format(self.default_optimizer_name(trainer.optimizer),
                                                                             self.property_name))
        else:
            self.optimizer_name = self.default_optimizer_name(trainer.optimizer).lower()
        super(OptimP, self).accept(trainer)

    @property
    def value(self):
        if self.property_name == 'name':
            return self.optimizer_name
        else:
            param_nr = int(self.property_arg) if self.property_arg is not None else 0
            param_group = self._param_group(self.trainer.optimizer, param_nr)
            return param_group[self.property_name]

    @value.setter
    def value(self, new_value):
        if self.property_name == 'name':
            self.optimizer_name = new_value
        else:
            param_nr = int(self.property_arg) if self.property_arg is not None else 0
            param_group = self._param_group(self.trainer.optimizer, param_nr)
            param_group[self.property_name] = new_value
