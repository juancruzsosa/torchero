import json
import zipfile

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import torchero
from torchero.utils.mixins import DeviceMixin
from torchero import meters
from torchero import SupervisedTrainer

class InputDataset(Dataset):
    """ Simple Dataset wrapper
    to transform input before giving it
    to the dataloader
    """
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.ds[idx])

    def __len__(self):
        return len(self.ds)

class Model(DeviceMixin):
    """ Model class that wrap nn.Module models to add
    training, prediction, saving & loading capabilities
    """
    def __init__(self, model):
        """ Constructor

        Arguments:
            model (nn.Module): Model to be wrapped
        """
        super(Model, self).__init__()
        self.model = model
        self.trainer = None

    def compile(self, optimizer, loss, metrics, hparams={}, callbacks=[], val_metrics=None):
        """ Compile this model with a optimizer a loss and set of given metrics

        Arguments:
            optimizer (str or instance of torch.optim.Optimizer): Optimizer to train the model
            loss (str or instance of torch.nn.Module): Loss (criterion) to be minimized
            metrics (list or dict of `torchero.meters.BaseMeter`, optional): A list of metrics
                or dictionary of metrics names and meters to record for training set
            hparams (list or dict of `torchero.meters.BaseMeter`, optional): A list of meters
                or dictionary of metrics names and hyperparameters to record
            val_metrics (list or dict of `torchero.meters.BaseMeter`, optional): Same as metrics argument
                for only used for validation set. If None it uses the same metrics as `metrics` argument.
            callbacks (list of `torchero.callbacks.Callback`): List of callbacks to use in trainings
        """
        self.trainer = SupervisedTrainer(model=self.model,
                                         criterion=loss,
                                         optimizer=optimizer,
                                         callbacks=callbacks,
                                         acc_meters=metrics,
                                         val_acc_meters=val_metrics,
                                         hparams=hparams)
        self.trainer.to(self.device)
        return self

    def input_to_tensor(self, *X):
        """ Converts inputs to tensors
        """
        return X

    def _predict_batch(self, *X):
        """ Generate output predictions for the input tensors

        This method can be called with a single input or multiple (If the model has multiple inputs)

        This method is not intended to be used directly. Use predict instead
        """
        self.model.train(False)
        with torch.no_grad():
            # Converts each input tensor to the given device
            X = list(map(self._convert_tensor, X))
            return self.model(*X)

    def to(self, device):
        """ Moves the model to the given device

        Arguments:
            device (str or torch.device)
        """
        super(Model, self).to(device)
        if self.trainer is not None:
            self.trainer.to(device)

    def _combine_preds(self, preds):
        """ Combines the list of predictions in a single tensor
        """
        return torch.stack(preds).cpu()

    def predict_on_dataloader(self, dl, has_targets=True):
        """ Generate output predictions on an dataloader

        Arguments:
            dl (`torch.utils.data.DataLoader`): input DataLoader
            has_targets (`torch.utils.DataLoader`): Omit target

        Notes:
            * The dataloader batches should yield `torch.Tensor`'s
        """
        preds = []
        for X in dl:
            if has_targets:
                X, y = X
            if isinstance(X, tuple):
                y = self._predict_batch(*X)
            else:
                y = self._predict_batch(X)
            preds.extend(y)
        preds = self._combine_preds(preds)
        return preds

    def predict(self,
                ds,
                batch_size=None,
                to_tensor=True,
                has_targets=False):
        """ Generate output predictions

        Arguments:
            ds (* `torch.utils.data.Dataset`
                * `torch.utils.data.DataLoader`
                * `list`
                * `np.array`): Input samples
            batch_size (int or None): Number of samples per batch. If None is
            passed it will default to 32.
            to_tensor (bool): Set this to True to convert inputs to tensors first (default behaviour)
            has_targets (bool): Whether to omit samples that already contains targets
        """
        dl = self._get_dataloader(ds,
                                  shuffle=False,
                                  batch_size=batch_size,
                                  shallow_dl=to_tensor)
        return self.predict_on_dataloader(dl, has_targets=has_targets)

    def train_on_dataloader(self, train_dl, val_dl=None, epochs=1):
        """ Trains the model for a fixed number of epochs

        Arguments:
            train_ds (`torch.utils.data.DataLoader`): Train dataloader
            val_ds (`torch.utils.data.Dataset`): Test dataloader
            epochs (int): Number of epochs to train the model
        """
        if self.trainer is None:
            raise Exception("Model hasn't been compiled with any trainer. Use model.compile first")

        self.trainer.train(dataloader=train_dl,
                           valid_dataloader=val_dl,
                           epochs=epochs)
        return self.trainer.history

    def evaluate_on_dataloader(self,
                               dataloader,
                               metrics=None):
        """ Evaluate metrics on a given dataloader

        Arguments:
            dataloader (`torch.utils.data.DataLoader`): Input Dataloader
            metrics (list of mapping, optional): Metrics to evaluate. If None is passed
            it will used the same defined at compile step
        """
        return self.trainer.evaluate(dataloader=dataloader,
                              metrics=metrics)


    def _create_dataloader(self, *args, **kwargs):
        return DataLoader(*args, **kwargs)

    def _get_dataloader(self,
                        ds,
                        batch_size=None,
                        shuffle=True,
                        collate_fn=None,
                        sampler=None,
                        shallow_dl=False):
        if isinstance(ds, (Dataset, list)):
            dl = self._create_dataloader(InputDataset(ds, self.input_to_tensor) if shallow_dl else ds,
                                         batch_size=batch_size or 32,
                                         shuffle=shuffle,
                                         collate_fn=collate_fn,
                                         sampler=sampler)
        elif isinstance(ds, DataLoader):
            dl = ds
        else:
            raise TypeError("ds type not supported. Use Dataloader or Dataset instances")
        return dl

    def evaluate(self,
                 ds,
                 metrics=None,
                 batch_size=None,
                 collate_fn=None,
                 sampler=None):
        """ Evaluate metrics

        Arguments:
            ds (* `torch.utils.data.Dataset`
                * `torch.utils.data.DataLoader`
                * `list`
                * `np.array`): Input data
            metrics (list of mapping, optional): Metrics to evaluate. If None is passed
                it will used the same defined at compile step
            batch_size (int or None): Number of samples per batch. If None is
                passed it will default to 32. Only relevant for non dataloader data
            collate_fn (callable, optional): merges a list of samples to form a
                mini-batch of Tensor(s). Used when using batched loading from a
                map-style dataset. See `torch.utils.data.DataLoader`
            sampler (Sampler or Iterable, optional): Defines the strategy to draw
                samples from the dataset. Can be any ``Iterable`` with ``__len__``
                implemented. If specified, :attr:`shuffle` must not be specified.
                See ``torch.utisl.data.DataLoader``
        """
        dl = self._get_dataloader(ds,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  collate_fn=collate_fn,
                                  sampler=sampler)
        return self.evaluate_on_dataloader(dl, metrics=metrics)

    def fit(self,
            train_ds,
            val_ds=None,
            epochs=1,
            batch_size=None,
            shuffle=True,
            collate_fn=None,
            sampler=None):
        """ Trains the model for a fixed number of epochs

        Arguments:
            train_ds (* `torch.utils.data.Dataset`
                * `torch.utils.data.DataLoader`
                * `list`
                * `np.array`): Train data
            val_ds (* `torch.utils.data.Dataset`
                * `torch.utils.data.DataLoader`
                * `list`
                * `np.array`): Validation data
            batch_size (int or None): Number of samples per batch. If None is
                passed it will default to 32. Only relevant for non dataloader data
            epochs (int): Number of epochs to train the model
            shuffle (bool): Set to ``True``to shuffle train dataset before every epoch. Only for
                non dataloader train data.
            collate_fn (callable, optional): merges a list of samples to form a
                mini-batch of Tensor(s). Used when using batched loading from a
                map-style dataset. See `torch.utils.data.DataLoader`
            sampler (Sampler or Iterable, optional): Defines the strategy to draw
                samples from the dataset. Can be any ``Iterable`` with ``__len__``
                implemented. If specified, :attr:`shuffle` must not be specified.
                See ``torch.utisl.data.DataLoader``
        """
        train_dl = self._get_dataloader(train_ds,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       collate_fn=collate_fn,
                                       sampler=sampler)
        if val_ds is None:
            val_dl = None
        else:
            val_dl = self._get_dataloader(val_ds,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          collate_fn=collate_fn,
                                          sampler=sampler)
        return self.train_on_dataloader(train_dl,
                                        val_dl,
                                        epochs)

    @property
    def config(self):
        return {'torchero_version': torchero.__version__}

    def save(self, path_or_fp):
        self.model.eval()
        with zipfile.ZipFile(path_or_fp, mode='w') as zip_fp:
            self._save_to_zip(zip_fp)

    def _save_to_zip(self, zip_fp):
        with zip_fp.open('model.pth', 'w') as fp:
            torch.save(self.model.state_dict(), fp)
        with zip_fp.open('config.json', 'w') as fp:
            fp.write(json.dumps(self.config, indent=4).encode())
        if self.trainer is not None:
            self.trainer._save_to_zip(zip_fp, prefix='trainer/')

    def load(self, path_or_fp):
        with zipfile.ZipFile(path_or_fp, mode='r') as zip_fp:
            self._load_from_zip(zip_fp)

    def _load_from_zip(self, zip_fp):
        with zip_fp.open('model.pth', 'r') as fp:
            self.model.load_state_dict(torch.load(fp))

class BinaryClassificationModel(Model):
    """ Model Class for Binary Classification (single or multilabel) tasks
    """
    def __init__(self, model, use_logits=True, threshold=0.5):
        """ Constructor

        Arguments:
            model (nn.Module): Model to be wrapped
            use_logits (bool): Set this as `True` if your model does **not**
                contain sigmoid as activation in the final layer (preferable)
                or 'False' otherwise
            threshold (float): Threshold used for metrics and predictions to determine if a prediction is true
        """
        super(BinaryClassificationModel, self).__init__(model)
        self.use_logits = use_logits
        self.threshold = threshold

    def compile(self, optimizer, loss=None, metrics=None, hparams={}, callbacks=[], val_metrics=None):
        """ Compile this model with a optimizer a loss and set of given metrics

        Arguments:
            optimizer (str or instance of torch.optim.Optimizer): Optimizer to train the model
            loss (str or instance of torch.nn.Module, optional): Loss (criterion) to be minimized.
                By default 'binary_cross_entropy_wl' (logits are already calculated on the loss)
                if use_entropy else 'binary_cross_entropy' (logits are not calculated on the loss)
            metrics (list or dict of `torchero.meters.BaseMeter`): A list of metrics
                or dictionary of metrics names and meters to record for training set.
                By default ['accuracy', 'balanced_accuracy']
            hparams (list or dict of `torchero.meters.BaseMeter`, optional): A list of meters
                or dictionary of metrics names and hyperparameters to record
            val_metrics (list or dict of `torchero.meters.BaseMeter`, optional): Same as metrics argument
                for only used for validation set. If None it uses the same metrics as `metrics` argument.
            callbacks (list of `torchero.callbacks.Callback`): List of callbacks to use in trainings
        """
        if loss is None:
            loss = 'binary_cross_entropy_wl' if self.use_logits else 'binary_cross_entropy'
        if metrics is None:
            metrics = ([meters.BinaryWithLogitsAccuracy(threshold=self.threshold),
                        meters.Recall(threshold=self.threshold, with_logits=True),
                        meters.Precision(threshold=self.threshold, with_logits=True),
                        meters.F1Score(threshold=self.threshold, with_logits=True)]
                       if self.use_logits else
                       [meters.BinaryAccuracy(threshold=self.threshold),
                        meters.Recall(threshold=self.threshold, with_logits=False),
                        meters.Precision(threshold=self.threshold, with_logits=False),
                        meters.F1Score(threshold=self.threshold, with_logits=False)])
        return super(BinaryClassificationModel, self).compile(optimizer=optimizer,
                                                              loss=loss,
                                                              metrics=metrics,
                                                              hparams=hparams,
                                                              callbacks=callbacks,
                                                              val_metrics=val_metrics)

    def _predict_batch(self, *X, output_probas=True):
        preds = super(BinaryClassificationModel, self)._predict_batch(*X)
        if self.use_logits:
            preds = nn.functional.sigmoid(preds)
        if not output_probas:
            preds = preds > self.threshold
        return preds

class ClassificationModel(Model):
    """ Model Class for Classification (for categorical targets) tasks
    """
    def __init__(self, model, use_softmax=True):
        """ Constructor

        Arguments:
            model (nn.Module): Model to be wrapped
            use_softmax (bool): Set this as `True` if your model does **not**
                contain softmax as activation in the final layer (preferable)
                or 'False' otherwise
        """
        super(ClassificationModel, self).__init__(model)
        self.use_softmax = use_softmax

    def compile(self, optimizer, loss=None, metrics=None, hparams={}, callbacks=[], val_metrics=None):
        """ Compile this model with a optimizer a loss and set of given metrics

        Arguments:
            optimizer (str or instance of torch.optim.Optimizer): Optimizer to train the model
            loss (str or instance of torch.nn.Module, optional): Loss (criterion) to be minimized.
                By default 'cross_entropy' if use_entropy else 'nll'
            metrics (list or dict of `torchero.meters.BaseMeter`): A list of metrics
                or dictionary of metrics names and meters to record for training set.
                By default ['accuracy', 'balanced_accuracy']
            hparams (list or dict of `torchero.meters.BaseMeter`, optional): A list of meters
                or dictionary of metrics names and hyperparameters to record
            val_metrics (list or dict of `torchero.meters.BaseMeter`, optional): Same as metrics argument
                for only used for validation set. If None it uses the same metrics as `metrics` argument.
            callbacks (list of `torchero.callbacks.Callback`): List of callbacks to use in trainings
        """
        if loss is None:
            loss = 'cross_entropy' if self.use_softmax else 'nll'
        if metrics is None:
            metrics = [meters.CategoricalAccuracy(), meters.BalancedAccuracy()]
        return super(ClassificationModel, self).compile(optimizer=optimizer,
                                                        loss=loss,
                                                        metrics=metrics,
                                                        hparams=hparams,
                                                        callbacks=callbacks,
                                                        val_metrics=val_metrics)
    def _predict_batch(self, *X):
        preds = super(ClassificationModel, self)._predict_batch(*X)
        if self.use_softmax:
            preds = nn.functional.softmax(preds)
        return preds

class RegressionModel(Model):
    """ Model Class for regression tasks
    """
    def compile(self, optimizer, loss='mse', metrics=None, hparams={}, callbacks=[], val_metrics=None):
        """ Compile this model with a optimizer a loss and set of given metrics

        Arguments:
            optimizer (str or instance of torch.optim.Optimizer): Optimizer to train the model
            loss (str or instance of torch.nn.Module, optional): Loss (criterion) to be minimized. Default: 'mse'
            metrics (list or dict of `torchero.meters.BaseMeter`): A list of metrics
                or dictionary of metrics names and meters to record for training set.
                By default RMSE
            hparams (list or dict of `torchero.meters.BaseMeter`, optional): A list of meters
                or dictionary of metrics names and hyperparameters to record
            val_metrics (list or dict of `torchero.meters.BaseMeter`, optional): Same as metrics argument
                for only used for validation set. If None it uses the same metrics as `metrics` argument.
            callbacks (list of `torchero.callbacks.Callback`): List of callbacks to use in trainings
        """
        if metrics is None:
            metrics = [meters.RMSE()]
        return super(RegressionModel, self).compile(optimizer=optimizer,
                                                    loss=loss,
                                                    metrics=metrics,
                                                    hparams=hparams,
                                                    callbacks=callbacks,
                                                    val_metrics=val_metrics)
