class CudaMixin(object):
    def __init__(self):
        self._use_cuda = False
        self._device = None
        super(CudaMixin, self).__init__()

    def cuda(self, device=None):
        self._use_cuda = True
        self._device = device

    def cpu(self):
        self._use_cuda = False

    def _tensor_to_cuda(self, x):
        if self._use_cuda:
            x = x.cuda(self._device)
        return x
