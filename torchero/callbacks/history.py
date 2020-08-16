import os
from collections import defaultdict
from operator import itemgetter
from torchero.callbacks.base import Callback
from collections import Iterable


class History(Callback):
    """ Callback that record history of all training/validation metrics
    """
    def __init__(self):
        super(History, self).__init__()
        self.registry = HistoryManager()

    def on_log(self):
        self.registry.append(self.trainer.epochs_trained,
                             self.trainer.steps_trained,
                             self.trainer.metrics)


class HistoryManager(Callback):
    UNRECOGNIZED_LEVEL_MESSAGE = (
        "Unrecognized level {level}. Level parameter should be either 'epoch' "
        "or 'step'"
    )

    def __init__(self):
        self.records = []

    def __iter__(self):
        yield from self.records

    def __getitem__(self, idx):
        return self.records[idx]

    def __len__(self):
        return len(self.records)

    def append(self, epoch, step, metrics):
        self.records.append({'epoch': epoch,
                             'step': step,
                             **metrics})

    def to_dataframe(self, level='epoch'):
        """ Returns the metrics history as a DataFrame format

        Parameters:
            level (str): It could either 'epoch' or 'step'. 
            'epoch' returns only the metrics of each epoch
            whereas 'step' returns from all steps

        Returns:
            A pandas DataFrame with column epoch, step (only if level='step')
            and a column for each metric
        """
        try:
            import pandas as pd
        except ImportError:
            raise Exception("Install pandas (pip install pandas) to export to history to dataframe")

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

    def epoch_plot(self, monitor, from_epoch=0, ax=None, title=None):
        """ Plot monitor history values across epochs

        Arguments:
            monitor (str): Monitor to plot
            from_epoch (int): Starting epoch in the plot
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

        for monitor in monitors:
            values = defaultdict(float)

            for record in self.records:
                epoch = record['epoch']
                if monitor in record and epoch >= from_epoch:
                    values[epoch] = record[monitor]

            ax.plot(list(values.keys()),
                    list(values.values()),
                    label=monitor,
                    marker='x')
        ax.legend()
        ax.grid(axis='both')
        ax.set_xlabel("epochs")

        if title is not None:
            ax.set_title(str(title))

    def __str__(self):
        return str(os.linesep.join(map(str, self.records)))

    def __repr__(self):
        return str(self)
