from collections import Iterable

from torch import nn
from torch import optim

from torchero import meters
from functools import partial

INVALID_MODE_INFERENCE_MESSAGE = (
    "Could not infer mode from meter {meter}"
)

def get_default_mode(meter):
    if hasattr(meter.__class__, 'DEFAULT_MODE'):
        return getattr(meter.__class__, 'DEFAULT_MODE')
    else:
        raise Exception(INVALID_MODE_INFERENCE_MESSAGE
                        .format(meter=getattr(meter, 'name', meter.__class__.__name__)))



optimizers = {
    'asgd': optim.ASGD,
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'adam': optim.Adam,
    'adamax': optim.Adamax,
    'lbfgs': optim.LBFGS,
    'rmsprop': optim.RMSprop,
    'rprop': optim.Rprop,
    'sgd': lambda params: optim.SGD(params, lr=1e-2),
    'sparseadam': optim.SparseAdam
}


def get_optimizer_by_name(name, model):
    if name not in optimizers:
        raise KeyError("Optimizer {} not found. "
                       "Optimizer availables: {}"
                       .format(repr(name),
                               ', '.join(map(repr, optimizers.keys()))))
    return optimizers[name](model.parameters())


losses = {
    'l1': nn.L1Loss,
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
        raise KeyError("Loss {} not found. Losses available: {}"
                       .format(repr(name),
                               ', '.join(map(repr, losses.keys()))))
    return losses[name]()


meters_by_name = {
    'mse': meters.MSE,
    'rmse': meters.RMSE,
    'msle': meters.MSLE,
    'rmsle': meters.RMSLE,
    'categorical_accuracy': meters.CategoricalAccuracy,
    'categorical_accuracy_percentage': lambda: meters.CategoricalAccuracy() * 100.0,
    'binary_accuracy': meters.BinaryAccuracy,
    'binary_accuracy_percentage': lambda: meters.BinaryAccuracy() * 100,
    'binary_accuracy_wl': meters.BinaryWithLogitsAccuracy,
    'binary_accuracy_wl_percentage': lambda: meters.BinaryWithLogitsAccuracy() * 100,
    'confusion_matrix': meters.ConfusionMatrix,
    'confusion_matrix_percentage': lambda: meters.ConfusionMatrix() * 100,
    'balanced_accuracy': meters.BalancedAccuracy,
}

for name, metric in (('recall', meters.Recall),
                     ('precision', meters.Precision),
                     ('npv', meters.NPV),
                     ('specificity', meters.Specificity),
                     ('f1', meters.F1Score),
                     ('f2', meters.F2Score)):
    meters_by_name.update({
        name: metric,
        name + '_wl': partial(metric, with_logits=True)
    })
    for agg_name in ('micro', 'macro', 'weighted'):
        meters_by_name.update({
            agg_name + '_' + name: partial(metric, with_logits=False, agg=agg_name),
            agg_name + '_' + name + '_wl': partial(metric, with_logits=True, agg=agg_name)
        })

for name, speed_metric, pace_metric in (('batches', meters.BatchSpeed, meters.BatchPace),
                                        ('it', meters.IterSpeed, meters.IterPace)):
    for unit_abbr, unit in (('sec', 'second'),
                            ('min', 'minute')):
        meters_by_name.update({name + '/' + unit_abbr: partial(speed_metric, time_unit=unit),
                               unit_abbr + '/' + name.replace('batches', 'batch'): partial(pace_metric, time_unit=unit)})


def get_meters_by_name(name):
    if name not in meters_by_name:
        raise KeyError("Meter {} not found. Meters available: {}"
                       .format(repr(name),
                               ', '.join(map(repr, meters_by_name.keys()))))
    return meters_by_name[name]()


def parse_meters(meters):
    def to_small_case(obj):
        if hasattr(obj, 'name'):
            s = str(obj.name)
        else:
            name = obj.__class__.__name__
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
            obj = get_meters_by_name(obj)
        return to_small_case(obj)

    if isinstance(meters, dict):
        return {k: parse(v) for k, v in meters.items()}
    elif isinstance(meters, Iterable):
        return {parse_name(v): parse(v) for v in meters}
    else:
        raise Exception("Expected iterable meters")

time_units = {'hour': 60*60,
              'hours': 60*60,
              'minute': 60,
              'minutes': 60,
              'second': 1,
              'seconds': 1}

def parse_time_unit(time_unit):
    if isinstance(time_unit, (int, float)):
        return time_unit
    elif isinstance(time_unit, str) and time_unit in time_units:
        return time_units[time_unit]
    elif isinstance(time_unit, str):
        raise ValueError("Invalid time_unit reference!")
    else:
        raise TypeError("Invalid type for time_unit")
