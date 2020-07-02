import torch

def format_metric(value):
    if isinstance(value, float):
        if abs(value) > 1e-3:
            return '{:.3f}'.format(value)
        else:
            return '{:1.2e}'.format(value)
    elif isinstance(value, str):
        return str(value)
    elif isinstance(value, dict):
        return repr({k: format_metric(v) for k, v in value.items()})
    elif isinstance(value, list) or isinstance(value, tuple):
        return '[{}]'.format(', '.join(map(format_metric, value)))
    else:
        return str(value)
