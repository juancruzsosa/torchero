from torch import optim
from torch import nn
from torchtrainer import meters
from collections import Iterable

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

losses = {'l1': nn.L1Loss,
          'mse': nn.MSELoss,
          'cross_entropy': nn.CrossEntropyLoss,
          'nll': nn.NLLLoss,
          'poisson_nll': nn.PoissonNLLLoss,
          'kl_div': nn.KLDivLoss,
          'binary_cross_entropy': nn.BCELoss,
          'binary_cross_entropy_wl': nn.BCEWithLogitsLoss,
          'margin_ranking': nn.MarginRankingLoss,
          'hinge': nn.HingeEmbeddingLoss,
          'multi_label_hinge': nn.MultiLabelMarginLoss,
          'smooth': nn.SmoothL1Loss,
          'soft_margin': nn.SoftMarginLoss,
          'multilabel_soft_margin': nn.MultiLabelSoftMarginLoss,
          'cosine': nn.CosineEmbeddingLoss,
          'multi_hinge': nn.MultiMarginLoss,
          'triplet_margin': nn.TripletMarginLoss
}

def get_loss_by_name(name):
    if name not in losses:
        raise KeyError("Loss {} not found. "
                       "Losses available: {}".format(repr(name),
                                                     ', '.join(map(repr, losses.keys()))))
    return losses[name]()

meters_by_name = {'mse': meters.MSE,
          'rmse': meters.RMSE,
          'msle': meters.MSLE,
          'rmsle': meters.RMSLE,
          'categorical_accuracy': meters.CategoricalAccuracy,
          'categorical_accuracy_percentage': lambda: meters.CategoricalAccuracy() * 100.0,
          'binary_accuracy': meters.BinaryAccuracy,
          'binary_accuracy_percentage': lambda: meters.BinaryAccuracy() * 100,
          'binary_with_logits_accuracy': meters.BinaryWithLogitsAccuracy,
          'binary_with_logits_accuracy_percentage': lambda: meters.BinaryWithLogitsAccuracy() * 100,
          'confusion_matrix': meters.ConfusionMatrix,
          'confusion_matrix_percentage': lambda: meters.ConfusionMatrix() * 100
}

def get_meters_by_name(name):
    if name not in meters_by_name:
        raise KeyError("Meter {} not found. "
                       "Meters available: {}".format(repr(name),
                                                     ', '.join(map(repr, meters_by_name.keys()))))
    return meters_by_name[name]()

def parse_meters(meters):
    def to_small_case(name):
        s = ''
        for i in range(len(name)-1):
            s += name[i].lower()
            if name[i].islower() and not name[i+1].islower():
                s += '_'
        s += name[-1].lower()
        return s

    def parse(obj):
        if isinstance(obj, str):
            return get_meters_by_name(obj)
        else:
            return obj

    def parse_name(obj):
        if isinstance(obj, str):
            return obj
        else:
            return to_small_case(obj.__class__.__name__)

    if isinstance(meters, dict):
        return {k: parse(v) for k, v in meters.items()}
    elif isinstance(meters, Iterable):
        return {parse_name(v): parse(v) for v in meters}
    else:
        raise Exception("Expected iterable meters")
