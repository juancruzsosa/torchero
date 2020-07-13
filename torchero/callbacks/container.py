from collections import deque

from torchero.callbacks.base import Callback


class CallbackContainer(Callback):
    """ Callback to group multiple callback
    """
    def __init__(self):
        super(CallbackContainer, self).__init__()
        self.callbacks = deque()

    def add(self, callback):
        callback.accept(self.trainer)
        self.callbacks.append(callback)

    def signal(self, selector_name):
        for callback in self.callbacks:
            callback_method = getattr(callback, selector_name)
            callback_method()

    def on_epoch_begin(self):
        self.signal('on_epoch_begin')

    def on_epoch_end(self):
        self.signal('on_epoch_end')

    def on_log(self):
        self.signal('on_log')

    def on_train_begin(self):
        self.signal('on_train_begin')

    def on_train_end(self):
        self.signal('on_train_end')
