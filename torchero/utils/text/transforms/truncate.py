import math
from abc import abstractmethod, ABCMeta

class Truncator(object, metaclass=ABCMeta):
    """ Truncates the input token sequence
    """
    def __init__(self, max_len):
        if max_len < 0:
            raise ValueError("max_len should be positive")
        self.max_len = max_len

    @abstractmethod
    def truncate(self, tokens):
        pass

    def __call__(self, tokens):
        if len(tokens) <= self.max_len:
            return tokens
        else:
            return self.truncate(tokens)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.max_len)

class LeftTruncator(Truncator):
    """ Truncates right part of the input token sequence
    """
    def truncate(self, tokens):
        return tokens[:self.max_len]

class RightTruncator(Truncator):
    """ Truncates left part of the input token sequence
    """
    def truncate(self, tokens):
        return tokens[-self.max_len:]

class CenterTruncator(Truncator):
    """ Truncates center part of the input token sequence
    """
    def truncate(self, tokens):
        mid = len(tokens)//2
        return tokens[mid-math.floor(self.max_len/2):mid+math.ceil(self.max_len/2)]
