from collections import deque
from .base import Hook

class HookContainer(Hook):
    def __init__(self, trainer):
        super(HookContainer, self).__init__()
        self.accept(trainer)
        self.hooks = deque()

    def attach(self, hook):
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