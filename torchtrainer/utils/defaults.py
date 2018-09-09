from torch import optim

def get_default_mode(meter):
    if hasattr(meter.__class__, 'DEFAULT_MODE'):
        return getattr(meter.__class__, 'DEFAULT_MODE')
    else:
        return ''

optimizers = {'asgd': optim.ASGD,
              'adadelta': optim.Adadelta,
              'adagrad': optim.Adagrad,
              'adam': optim.Adam,
              'adamax': optim.Adamax,
              'lbfgs': optim.LBFGS,
              'rmsprop': optim.RMSprop,
              'rprop': optim.Rprop,
              'sgd': lambda params: optim.SGD(params, lr=1e-2),
              'sparseadam': optim.SparseAdam}

def get_optimizer_by_name(name, model):
    if name not in optimizers:
        raise KeyError("Optimizer {} not found. "
                       "Optimizer availables: {}".format(repr(name),
                                                         ', '.join(map(repr, optimizers.keys()))))
    return optimizers[name](model.parameters())
