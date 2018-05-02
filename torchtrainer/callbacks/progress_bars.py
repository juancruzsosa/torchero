from .base import Callback
from .container import CallbackContainer

try:
    import tqdm
except ImportError:
    print("tqdm not installed!")
    print("install tqdm for progress bars support.")
    raise


class ProgbarLogger(Callback):
    """ Callback that displays progress bars to monitor training/validation metrics
    """

    def __init__(self, ascii=False, notebook=False):
        """ Constructor

        Arguments:

            ascii (`bool`): if true display progress bar in ASCII mode.
            notebook (`bool`): Make outputs compatible for python notebooks
        """
        self.ascii = ascii
        if notebook:
            self.tqdm = tqdm.tqdm_notebook
        else:
            self.tqdm = tqdm.tqdm

        self.step_tqdms = []
        self.step_bars = []

    def on_train_begin(self):
        self.epoch_tqdm = self.tqdm(total=self.trainer.total_epochs, unit='epoch', leave=True, ascii=self.ascii)
        self.epoch_bar = self.epoch_tqdm.__enter__()

    def on_epoch_begin(self):
        step_tqdm = self.tqdm(total=self.trainer.total_steps, unit=' batchs', leave=True, ascii=self.ascii)
        self.step_tqdms.append(step_tqdm)
        self.step_bars.append(step_tqdm.__enter__())


    def format(self, value):
        if isinstance(value, float):
            return '{:.3f}'.format(value)
        else:
            return str(value)

    def on_log(self):
        metrics = {name: self.format(value) for name, value in self.trainer.metrics.items()}
        step_bar = self.step_bars[-1]
        step_bar.set_postfix(**metrics),
        step_bar.update(self.trainer.logging_frecuency)

    def on_epoch_end(self):
        self.epoch_bar.update()

    def on_train_end(self):
        self.epoch_bar.__exit__()
        self.epoch_tqdm.close()

        for step_tqdm, step_bar in zip(self.step_tqdms, self.step_bars):
            step_bar.__exit__()
            step_tqdm.close()

        self.step_tqdms = []
        self.step_bars = []
