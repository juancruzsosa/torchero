import logging
from collections import namedtuple
from itertools import chain
from multiprocessing import Pool

logger = logging.getLogger('compose')

class Compose(object):
    """ Composes several transforms for Text togheter
    """
    @classmethod
    def from_dict(cls, transforms):
        return cls(**transforms)

    def __init__(self, *args, **kwargs):
        step_names = ['step_{}' for i, _ in enumerate(args)]
        step_names += list(kwargs.keys())
        if 'fit' in step_names:
            raise ValueError("Invalid step_name 'fit'. Choose another name")
        self.step_names = step_names
        pipeline_class = namedtuple('pipeline', step_names)
        self.transforms = pipeline_class(**dict(zip(step_names, list(args) + list(kwargs.values()))))

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def fit(self, xs):
        """ Fit every transform with the given input data in order
        """
        for i, t in enumerate(self.transforms):
            if hasattr(t, 'fit'):
                logger.debug("[{i}] Fitting {t}".format(i=i, t=t))
                t.fit(xs)
            if i < len(self.transforms)-1:
                logger.debug("[{i}] Transforming {t}".format(i=i, t=t))
                xs = list(map(t, xs))

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for step_name, t in zip(self.step_names, self.transforms):
            format_string += '\n'
            format_string += '    {}={},'.format(step_name, t)
        format_string += '\n)'
        return format_string

    def __getitem__(self, idx):
        return self.transforms[idx]

    def __getattr__(self, attr_name):
        return getattr(self.transforms, attr_name)

    def __getstate__(self):
        return dict(zip(self.step_names, self.transforms))

    def __setstate__(self, d):
        step_names, transforms = zip(*d.items())
        pipeline_class = namedtuple('pipeline', step_names)
        self.step_names = step_names
        self.transforms = pipeline_class(**dict(zip(step_names, transforms)))

    def __len__(self):
        return len(self.transforms)
