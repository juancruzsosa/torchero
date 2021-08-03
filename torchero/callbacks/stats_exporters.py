import os
from enum import Enum

from torchero.callbacks.base import Callback


class LogLevel(Enum):
    EPOCH = 'epoch'
    STEP = 'step'


class CSVLogger(Callback):
    """ Export the training statistics to csv file
    """

    UNRECOGNIZED_LEVEL = (
        "Unrecognized level {level}. Level parameter should be either 'epoch' "
        "or 'step'"
    )

    def __init__(self, output, append=False, columns=None, hparams_columns=None, level='epoch'):
        """ Constructor

        Arguemnts:
            output (str): Name of csv file to export
            append (bool): Append to file instead of overwriting it
            columns (list): List of columns name to export. If is none select
            hparams_columns (list): List of hyperparams to export
            to display all columns (default)
        """
        self.output = output
        self.append = append
        self.columns = columns
        self.hparams_columns = hparams_columns
        try:
            self.level = LogLevel(level)
        except ValueError:
            raise ValueError(self.UNRECOGNIZED_LEVEL.format(level=repr(level)))

    def on_train_begin(self):
        """ Creates & opens the file
        """
        if self.columns is None:
            extra_cols = ['epoch']
            if self.level is LogLevel.STEP:
                extra_cols.append('step')
            self.columns = extra_cols + self.trainer.meters_names()

        if self.hparams_columns is None:
            self.hparams_columns = list(self.trainer.hparams.keys())

        if os.path.isfile(self.output) and self.append:
            new_file = True
            mode = 'a+'
        else:
            new_file = False
            mode = 'w+'

        self.file_handle = open(self.output, mode)

        if not new_file:
            self.file_handle.write(','.join(self.columns + self.hparams_columns))

    def _write_line(self):
        """ Write the metrics to a new line
        """
        if len(self.trainer.metrics) == 0:
            return

        stats = self.trainer.metrics
        stats['epoch'] = self.trainer.epochs_trained
        stats['step'] = self.trainer.steps_trained

        hparams = self.trainer.hparams

        new_row = [stats.get(column, '') for column in self.columns]
        new_row.extend([hparams.get(column, '') for column in self.hparams_columns])

        self.file_handle.write(os.linesep + ','.join(map(str, new_row)))
        self.file_handle.flush()

    def on_log(self):
        """ Add a new line in the file if its configured per step
        """
        if self.level is LogLevel.STEP:
            self._write_line()

    def on_epoch_end(self):
        """ Add a new line in the file if its configured per epoch
        """
        if self.level is LogLevel.EPOCH:
            self._write_line()

    def on_train_end(self):
        """ Closes the files
        """
        self.file_handle.close()
        del self.file_handle

    def __repr__(self):
        columns = ""
        if self.columns is not None:
            columns = ", columns={}".format(repr(self.columns))
        hparams_columns = ""
        if self.hparams_columns is not None:
            hparams_columns = ", hparams_columns={}".format(repr(self.hparams_columns))
        return "{cls}(output={output}, append={append}, level={level}{columns}{hparams})".format(
            cls=self.__class__.__name__,
            output=repr(self.output),
            append=repr(self.append),
            level=repr(self.level.name.lower()),
            columns=columns,
            hparams=hparams_columns
        )
