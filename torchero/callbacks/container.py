from collections import deque

from torchero.callbacks.base import Callback


class CallbackContainer(Callback):
    """ Callback to group multiple callbacks
    """
    def __init__(self):
        super(CallbackContainer, self).__init__()
        self.callbacks = deque()

    def add(self, callback):
        """ Add a new callback to the list
        """
        callback.accept(self.trainer)
        self.callbacks.append(callback)

    def signal(self, selector_name):
        """ Signal an event to each callback
        """
        for callback in self.callbacks:
            callback_method = getattr(callback, selector_name)
            callback_method()

    def on_epoch_begin(self):
        """ Broadcast on_epoch_begin signal to each callback
        """
        self.signal('on_epoch_begin')

    def on_epoch_end(self):
        """ Broadcast on_epoch_end signal to each callback
        """
        self.signal('on_epoch_end')

    def on_log(self):
        """ Broadcast on_log signal to each callback
        """
        self.signal('on_log')

    def on_train_begin(self):
        """ Broadcast on_train_begin signal to each callback
        """
        self.signal('on_train_begin')

    def on_train_end(self):
        """ Broadcast on_train_end signal to each callback
        """
        self.signal('on_train_end')

    def __repr__(self):
        return repr(list(self.callbacks))

    def __iter__(self):
        """ Iterates over the list of callbacks
        """
        return iter(self.callbacks)

    def __len__(self):
        """ Returns the number of callbacks
        """
        return len(self.callbacks)
