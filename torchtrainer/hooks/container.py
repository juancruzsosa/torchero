from collections import deque
from .base import Hook

class HookContainer(Hook):
    """ Hook to group multiple hook
    """
    def __init__(self):
        super(HookContainer, self).__init__()
        self.hooks = deque()

    def add(self, hook):
        hook.accept(self.trainer)
        self.hooks.append(hook)

    def signal(self, selector_name):
        for hook in self.hooks:
            hook_method = getattr(hook, selector_name)
            hook_method()

    def pre_epoch(self):
        self.signal('pre_epoch')

    def post_epoch(self):
        self.signal('post_epoch')

    def log(self):
        self.signal('log')

    def pre_training(self):
        self.signal('pre_training')

    def post_training(self):
        self.signal('post_training')
