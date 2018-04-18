from .base import Hook
from .container import HookContainer

try:
    import tqdm
except ImportError:
    print("tqdm not installed!")
    print("install tqdm for progress bars support.")
    raise


class ProgressBars(Hook):
    def __init__(self, ascii=False, notebook=False):
        self.ascii = ascii
        if notebook:
            self.tqdm = tqdm.tqdm_notebook
        else:
            self.tqdm = tqdm.tqdm

        self.step_tqdms = []
        self.step_bars = []

    def pre_training(self):
        self.epoch_tqdm = self.tqdm(total=self.trainer.total_epochs, unit='epoch', leave=True, ascii=self.ascii)
        self.epoch_bar = self.epoch_tqdm.__enter__()

    def pre_epoch(self):
        step_tqdm = self.tqdm(total=self.trainer.total_steps, unit=' batchs', leave=True, ascii=self.ascii)
        self.step_tqdms.append(step_tqdm)
        self.step_bars.append(step_tqdm.__enter__())


    def format(self, value):
        if isinstance(value, float):
            return '{:.3f}'.format(value)
        else:
            return str(value)

    def log(self):
        last_stats = {name: self.format(value) for name, value in self.trainer.last_stats.items()}
        step_bar = self.step_bars[-1]
        step_bar.set_postfix(**last_stats),
        step_bar.update(self.trainer.logging_frecuency)

    def post_epoch(self):
        self.epoch_bar.update()

    def post_training(self):
        self.epoch_bar.__exit__()
        self.epoch_tqdm.close()

        for step_tqdm, step_bar in zip(self.step_tqdms, self.step_bars):
            step_bar.__exit__()
            step_tqdm.close()

        self.step_tqdms = []
        self.step_bars = []
