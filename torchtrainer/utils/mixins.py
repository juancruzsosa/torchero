import torch


class CudaMixin(object):
    def __init__(self):
        self._use_cuda = False
        super(CudaMixin, self).__init__()

    def cuda(self):
        self._use_cuda = True

    def cpu(self):
        self._use_cuda = False

    def _tensor_to_cuda(self, x):
        if self._use_cuda:
            x = x.cuda()
        return x
