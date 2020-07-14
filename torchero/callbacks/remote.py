import json
import warnings

from torchero.callbacks.base import Callback

try:
    import requests
except ImportError:
    requests = None


class RemoteMonitor(Callback):
    """ Keras RemoteMonitor clone to stream events to a server.
    """
    def __init__(self,
                 root='http://localhost:9000',
                 path='/publish/epoch/end/',
                 field='data',
                 headers=None,
                 send_as_json=False,
                 monitors=None):
        """ Constructor

        Arguments:
            root (str): Root server url
            path (str): Relative path to root to post events
            field (str): Json field of post data
            send_as_json (bool): If false sends data as plain json. Otherwise
            sends as json.
            headers (str): Http headers
            monitors (list): List of monitors names to include in sended data.
        """
        super(RemoteMonitor, self).__init__()

        if requests is None:
            raise ImportError("RemoteMonitor requires "
                              "the 'requests' library.")
        self.root = root
        self.path = path
        self.field = field
        self.headers = headers
        self.monitors = monitors
        self.send_as_json = send_as_json

    def on_epoch_end(self):
        monitors = self.monitors
        if monitors is None:
            monitors = self.trainer.metrics.keys()

        metrics = {name: self.trainer.metrics[name]
                   for name in monitors
                   if name in self.trainer.metrics}
        metrics.update({'epoch': self.trainer.epochs_trained})

        try:
            if self.send_as_json:
                requests.post(self.root + self.path,
                              json={self.field: metrics},
                              headers=self.headers)
            else:
                requests.post(self.root + self.path,
                              {self.field: json.dumps(metrics)},
                              headers=self.headers)
        except Exception:
            warnings.warn("Could not reach {}".format(self.root),
                          RuntimeWarning)
