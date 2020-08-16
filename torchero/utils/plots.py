import matplotlib.pyplot as plt

def smooth_curve(xs, alpha=0.0):
    res = []
    for x in xs:
        if len(res) == 0:
            res.append(x)
        else:
            res.append(res[-1] * (1 - alpha) + alpha * x)
    return res
