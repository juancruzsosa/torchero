from torch import optim
from torch import nn

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
