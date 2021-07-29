import math
import os
from itertools import product, chain
from collections import defaultdict
from operator import itemgetter
from collections import Iterable

import pandas as pd

from torchero.callbacks.base import Callback
from torchero.utils.plots import smooth_curve

class History(Callback):
    """ Callback to save the training history of all metrics
    """
    def __init__(self):
        super(History, self).__init__()
        self.registry = HistoryManager()

    def on_log(self):
        """ Append the new set of metrics of the current epoch
        """
        self.registry.append(self.trainer.epochs_trained,
                             self.trainer.steps_trained,
                             self.trainer.metrics,
                             self.trainer.hparams)

    def __repr__(self):
        return "{cls}()".format(
            cls=self.__class__.__name__
        )


class HistoryManager(Callback):
    """ Manages the training history
    """
    UNRECOGNIZED_LEVEL_MESSAGE = (
        "Unrecognized level {level}. Level parameter should be either 'epoch' "
        "or 'step'"
    )

    def __init__(self):
        """ Initializes the instance with a blank history
        """
        self.records = []

    def __iter__(self):
        """ Returns an iterator of all the records
        """
        yield from self.records

    def __getitem__(self, idx):
        """ Access to the metrics & hyper-parameters of a given time-step
        """
        return self.records[idx]

    def __len__(self):
        """ Returns the number of metrics & hyper-parameters recorded
        """
        return len(self.records)

    def columns(self):
        """ Return the names of all the metrics & hyper-parameters recorded
        """
        return set(chain(*map(lambda x: set(x.keys()), self.records)))

    def append(self, epoch, step, metrics, hparams):
        """ Add a new data point to the history
        """
        self.records.append({'epoch': epoch,
                             'step': step,
                             **metrics,
                             **hparams})

    def to_dataframe(self, level='epoch'):
        """ Returns this object as a DataFrame object

        Parameters:
            level (str): It could either 'epoch' or 'step'. 
            'epoch' returns only the metrics of each epoch
            whereas 'step' returns from all steps

        Returns:
            A pandas DataFrame with column epoch, step (only if level='step')
            and a column for each metric
        """

        df = (pd.DataFrame.from_records(self.records)
                .sort_values(['epoch', 'step']))
        if level == 'step':
            return df
        elif level == 'epoch':
            return (df.drop_duplicates(['epoch'], keep='last')
                      .drop(columns=['step'])
                      .reset_index(drop=True))
        else:
            raise ValueError(self.UNRECOGNIZED_LEVEL_MESSAGE.format(level=repr(level)))

    def step_plot(self, monitor, from_step=1, ax=None):
        """ Plot monitor history values across trained iterations

        Arguments:
            monitor (str): Monitor to plot
            from_step (int): Starting iteration in the plot
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        def condition(record):
            return monitor in record and record['step'] >= from_step

        x = map(itemgetter('step'), filter(condition, self.records))
        y = map(itemgetter(monitor), filter(condition, self.records))

        ax.plot(list(x), list(y), label=monitor)
        ax.legend()

    def epoch_plot(self, monitor, from_epoch=0, ax=None, title=None, ylabel=None, smooth=0):
        """ Plot monitor history values across epochs

        Arguments:
            monitor (list, str): Monitor to plot or list of monitor to plot
            from_epoch (int): Starting epoch in the plot
            ax (``matplotlib.axes.Axes``): Matplotlib axis to use for the plot.
            If None is passed uses the default axis
            title (str): Title of the plot or None for no title
            smooth (str): Smooth factor for the line plot.
            For smoothing it uses the Exponentially Weighted Average technique.
        """
        import matplotlib.pyplot as plt

        if isinstance(monitor, str):
            monitors = [monitor]
        elif isinstance(monitor, Iterable):
            monitors = list(monitor)
        else:
            raise TypeError("Monitor parameter should be either a string or a list of strings")


        if ax is None:
            ax = plt.gca()

        cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for plot_num, monitor in enumerate(monitors):
            values = defaultdict(float)

            for record in self.records:
                epoch = record['epoch']
                if monitor in record and epoch >= from_epoch:
                    values[epoch] = record[monitor]

            x = list(values.keys())
            y = list(values.values())
            if smooth > 0:
                ax.plot(x, y, alpha=0.2, color=cmap[plot_num])
                y = smooth_curve(y, alpha=1-smooth)

            ax.plot(x, y, label=monitor, color=cmap[plot_num])
        ax.legend()
        ax.grid(axis='both')
        ax.set_xlabel("epochs")
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(str(title))

    def plot(self,
             layout='default',
             figsize=None,
             from_epoch=0,
             title="Metrics",
             smooth=0):
        """ Plot a figure for a group of metrics

        Arguments:
            layout (str or dict): Plot layout. If a dict is passed
            it should have one entry per cell, keys are row and cell positions
            whereas values are a dictionary of options for that cell.
            If string is passed, options are:
                *) 'default' is a layout that has one cell per metric
                comparing train/validation in the same plot
                *) 'columns': has a column per set (train/validation).
            figsize (tuple or int): Matplotlib figure size.
            from_epoch (int): Start epoch for all plot
            title (str): Plot header's title
            smooth (float): Amount of smoothing for all lineplots

        Example:
            For a Figure with losses (left plot) and accuracies (right plot)

            >>> layout = {
                (0, 0): {'metrics': ['train_loss', 'val_loss'], 'title': 'Losses', 'smooth': 0.5, 'ylabel': 'loss'},
                (0, 1): {'metrics': ['train_acc', 'val_acc'], 'title': 'Accuracy', 'ylabel': 'loss'}
            }
            >>> trainer.history.plot(layout, title="Training Results", figsize=(20, 10))
        """

        import matplotlib.pyplot as plt

        if isinstance(layout, str):
            if layout == 'default':
                layout = self.get_default_layout()
            elif layout == 'column':
                layout = self.get_column_layout()
            else:
                raise ValueError("Unknown layout '{}'. Use 'default', 'columns' or a dictionary".format(layout))

        nrows = max(map(lambda x: x[0], layout.keys()))+1
        ncols = max(map(lambda x: x[1], layout.keys()))+1

        if figsize is None:
            figsize = (15, 10 * nrows / ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axs = axs.reshape((nrows, ncols))

        for (i, j), item in layout.items():
            self.epoch_plot(item['metrics'],
                            title=item.get('title', ''),
                            ylabel=item.get('ylabel', ''),
                            ax=axs[i][j],
                            from_epoch=item.get('from_epoch', from_epoch),
                            smooth=item.get('smooth', smooth))

        if title is not None:
            plt.suptitle(title)
        return fig, axs

    def get_default_layout(self):
        """ Returns the configuration layout that compares the training and validation results
            of each metric in the cell of the plot tile
        """
        cols = self.columns()-{'epoch', 'step'}
        suffixes = set(x.split('_', maxsplit=1)[1] if '_' in x else x for x in cols)
        suffixes = [(s, sorted([c for c in cols if c.endswith(s)])) for s in suffixes]
        suffixes = sorted(suffixes, key=lambda x: x[0])
        n = len(suffixes)
        nrows = math.ceil(math.sqrt(n))
        ncols = math.ceil(n / nrows)
        indices = product(range(nrows), range(ncols))
        layout = {(i, j): {'metrics': cols,
                           'title': '/'.join(cols),
                           'ylabel': title}
                  for (i, j), (title, cols) in zip(indices, suffixes)}

        return layout

    def get_column_layout(self):
        """ Returns the configuration layout that distributes the metric j for the data-set k
        at row k and column j
        """
        cols = self.columns()-{'epoch', 'step'}
        prefixes = set(x.split('_', maxsplit=1)[0] for x in cols)
        prefixes = [(p, sorted([c for c in cols if c.startswith(p)])) for p in prefixes]
        prefixes = sorted(prefixes, key=lambda x: x[0])

        layout = {(i, j): {'metrics': col, 'title': col, 'ylabel': col[len(p)+1:]}
                    for j, (p, cols) in enumerate(prefixes)
                        for i, col in enumerate(cols)}
        return layout


    def __str__(self):
        return str(os.linesep.join(map(str, self.records)))

    def __repr__(self):
        return str(self)
