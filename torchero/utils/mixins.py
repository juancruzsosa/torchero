import torch
from torch.autograd import Variable

class DeviceMixin(object):
    def __init__(self):
        self._device = None
        super(DeviceMixin, self).__init__()

    def cuda(self, device=None):
        """ Turn to gpu

        Arguments:
            device (str or torch.device): torch.device or Cuda device name
        """
        if device is not None:
            if torch.device(device).type != 'cuda':
                raise RuntimeError("Invalid device, must be cuda device")
        else:
            device = 'cuda'
        self.to(device)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self.to(device)

    def to(self, device):
        """ Turns to Device

        Arguments:
            device (str or torch.device): torch.device or Cuda device name
        """
        if device is None:
            self._device = None
        else:
            device = torch.device(device)
            if (device.type == 'cuda') and (not torch.cuda.is_available()):
                raise RuntimeError("no CUDA-capable device is detected")
            self._device = device

    def cpu(self):
        """ Turn model to cpu
        """
        self.to('cpu')

    def _convert_tensor(self, x):
        if (self._device is not None) and (torch.get_device(x) != self._device):
            x = x.to(self._device)
        return x

    def _prepare_tensor(self, x):
        if torch.is_tensor(x):
            return Variable(self._convert_tensor(x))
        elif isinstance(x, (tuple, list)):
            return type(x)([self._prepare_tensor(t) for t in x])
        else:
            return x
