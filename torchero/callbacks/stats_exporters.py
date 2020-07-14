import os
from enum import Enum

from torchero.callbacks.base import Callback


class LogLevel(Enum):
    EPOCH = 1
    STEP = 0


class CSVLogger(Callback):
    """ Export training statistics to csv file
    """
    UNRECOGNIZED_LEVEL = (
        "Unrecognized level {level}. Level parameter should be either 'epoch' "
        "or 'step'"
    )

    def __init__(self, output, append=False, columns=None, level='epoch'):
        """ Constructor

        Arguemnts:
            output (str): Name of csv file to export
            append (bool): Append to file instead of overwriting it
            columns (list): List of columns name to export. If is none select
            to display all columns (default)
        """
        self.output = output
        self.append = append
        self.columns = columns
        if level == 'epoch':
            self.level = LogLevel.EPOCH
        elif level == 'step':
            self.level = LogLevel.STEP
        else:
            raise ValueError(self.UNRECOGNIZED_LEVEL.format(level=repr(level)))

    def on_train_begin(self):
        if self.columns is None:
            extra_cols = ['epoch']
            if self.level is LogLevel.STEP:
                extra_cols.append('step')
            self.columns = extra_cols + self.trainer.meters_names()

        if os.path.isfile(self.output) and self.append:
            new_file = True
            mode = 'a+'
        else:
            new_file = False
            mode = 'w+'

        self.file_handle = open(self.output, mode)

        if not new_file:
            self.file_handle.write(','.join(self.columns))

    def _write_line(self):
        if len(self.trainer.metrics) == 0:
            return

        stats = self.trainer.metrics
        stats['epoch'] = self.trainer.epochs_trained
        stats['step'] = self.trainer.steps_trained

        new_row = (stats.get(column, '') for column in self.columns)

        self.file_handle.write(os.linesep + ','.join(map(str, new_row)))
        self.file_handle.flush()

    def on_log(self):
        if self.level is LogLevel.STEP:
            self._write_line()

    def on_epoch_end(self):
        if self.level is LogLevel.EPOCH:
            self._write_line()

    def on_train_end(self):
        self.file_handle.close()
