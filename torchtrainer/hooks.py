from collections import deque

class Hook(object):
    """ Hooks for epoch/batch training events
    """
    def __init__(self):
        """ Constructor
        """
        self.trainer = None

    def accept(self, trainer):
        """ Accept a trainer

        Args:
            trainer(instance of :class:`torchtrainer.base.BaseTrainer`): Trainer to attach to
        """
        self.trainer = trainer

    def pre_epoch(self):
        """ Trigger called before every epoch
        """
        pass

    def post_epoch(self):
        """ Trigger called after every epoch
        """
        pass

    def log(self):
        """ Trigger called after every log update
        """
        pass

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

