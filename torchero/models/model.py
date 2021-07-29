import json
import zipfile
import importlib
from functools import partial

import numpy as np

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

class ModelImportException(Exception):
    pass

class PredictionItem(object):
    def __init__(self, preds):
        self._preds = preds

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               repr(self._preds))

    @property
    def tensor(self):
        return self._preds

    def numpy(self):
        return self._preds.cpu().numpy()

class PredictionsResult(object):
    def __init__(self, preds, pred_class=PredictionItem):
        self._preds = [
            pred_class(pred) for pred in preds
        ]

    @property
    def tensor(self):
        return torch.stack([pred.tensor for pred in self._preds])

    def numpy(self):
        return np.stack([pred.numpy() for pred in self._preds])

    def __iter__(self):
        return iter(self._preds)

    def __len__(self):
        return len(self._preds)

    def __getitem__(self, idx):
        return self._preds[idx]

class Model(DeviceMixin):
    """ Model Class for Binary Classification (single or multilabel) tasks
    """
    """ Model class that wrap nn.Module models to add
    training, prediction, saving & loading capabilities
    """

    @classmethod
    def load_from_file(_, path_or_fp, net=None):
        """ Load a saved model from disk an convert it to the desired type (ImageModel, TextModel, etc)

        Arguments:
            net (nn.Module): Neural network initialized in the same way as the saved one.
            path_or_fp (file-like or str): Path to saved model
        """
        with zipfile.ZipFile(path_or_fp, mode='r') as zip_fp:
            with zip_fp.open('config.json', 'r') as fp:
                config = json.loads(fp.read().decode('utf-8'))
        model_type = config['torchero_model_type']
        module = importlib.import_module(model_type['module'])
        model_type = getattr(module, model_type['type'])
        print(model_type)
        if net is None:
            if 'net' not in config:
                raise ModelImportException("Invalid network configuration json (Expected 'net' key)")
            net_type = config['net']['type']
            net_module = importlib.import_module(net_type['module'])
            net_type = getattr(net_module, net_type['type'])
            print(net_type)
            if 'config' not in config['net']:
                raise ModelImportException("Network configuration not found in config.json ('net.config'). Create function passing an already initialized network")
            if hasattr(net_type, 'from_config') and 'config' in config['net']:
                net = net_type.from_config(config['net']['config'])
        model = model_type(net)
        model.load(path_or_fp)
        return model

    def __init__(self, model):
        """ Constructor

        Arguments:
            model (nn.Module): Model to be wrapped
        """
        super(Model, self).__init__()
        self.model = model
        self.trainer = None

    def pred_class(self, preds):
        return PredictionsResult(preds)


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
        preds = torch.stack(preds)
        return self.pred_class(preds)

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
        config = {
            'torchero_version': torchero.__version__,
            'torchero_model_type': {'module': self.__class__.__module__,
                                    'type': self.__class__.__name__},
            'compiled': self.trainer is not None,
        }
        if hasattr(self.model, 'config'):
            config.update({'net': {
                'type': {'module': self.model.__class__.__module__,
                         'type': self.model.__class__.__name__},
                'config': self.model.config
            }})
        return config

    def init_from_config(self, config):
        pass

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
        with zip_fp.open('config.json', 'r') as config_fp:
            config = json.loads(config_fp.read().decode())
        if config['compiled'] is True:
            self.trainer = SupervisedTrainer(model=self.model,
                                             criterion=None,
                                             optimizer=None)
            self.trainer._load_from_zip(zip_fp, prefix='trainer/')
        self.init_from_config(config)

class UnamedClassificationPredictionItem(PredictionItem):
    """ Model Prediction with classes names
    """
    def __init__(self, preds):
        super(UnamedClassificationPredictionItem, self).__init__(preds)

    def as_dict(self):
        return dict(enumerate(self._preds.tolist()))

    def max(self):
        return self._preds.max().item()

    def argmax(self):
        return self._preds.argmax().item()

    def topk(self, k):
        values, indices = self._preds.topk(k)
        return list(zip(indices.tolist(), values.tolist()))

    def as_tuple(self):
        return tuple(pred.tolist())

class NamedClassificationPredictionItem(PredictionItem):
    """ Model Prediction with classes names
    """
    def __init__(self, preds, names=None):
        super(NamedClassificationPredictionItem, self).__init__(preds)
        self.names = names

    def max(self):
        return self._preds.max().item()

    def argmax(self):
        return self.names[self._preds.argmax().item()]

    def topk(self, k):
        values, indices = self._preds.topk(k)
        names = map(self.names.__getitem__, indices.tolist())
        return list(zip(names, values.tolist()))

    def as_dict(self):
        return dict(zip(self.names, self._preds.tolist()))

    def as_tuple(self):
        return tuple(pred.tolist())

class ClassificationPredictionsResult(PredictionsResult):
    """ List of model classification predictions
    """
    def __init__(self, preds, names=None):
        self.names = names
        if self.names is None:
            pred_class = UnamedClassificationPredictionItem
        else:
            pred_class = partial(NamedClassificationPredictionItem, names=self.names)
        super(ClassificationPredictionsResult, self).__init__(preds, pred_class=pred_class)

    def as_dict(self):
        return [pred.as_dict() for pred in self._preds]

    def as_tuple(self):
        return [pred.as_tuple() for pred in self._preds]

    def max(self):
        return [pred.max() for pred in self._preds]

    def argmax(self):
        return [pred.argmax() for pred in self._preds]

    def topk(self, k):
        return [pred.topk(k) for pred in self._preds]

    def as_df(self):
        import pandas as pd
        return pd.DataFrame.from_records(self.as_dict())

class BinaryClassificationModel(Model):
    def __init__(self, model, use_logits=True, threshold=0.5, labels=None):
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
        self.labels = labels

    @property
    def config(self):
        config = super(BinaryClassificationModel, self).config
        config['labels'] = self.labels
        return config

    def init_from_config(self, config):
        super(BinaryClassificationModel, self).init_from_config(config)
        self.labels = config['labels']

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

    def pred_class(self, preds):
        return ClassificationPredictionsResult(preds, names=self.labels)

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
    def __init__(self, model, use_softmax=True, classes=None):
        """ Constructor

        Arguments:
            model (nn.Module): Model to be wrapped
            use_softmax (bool): Set this as `True` if your model does **not**
                contain softmax as activation in the final layer (preferable)
                or 'False' otherwise
        """
        super(ClassificationModel, self).__init__(model)
        self.use_softmax = use_softmax
        self.classes = classes

    @property
    def config(self):
        config = super(ClassificationModel, self).config
        config['classes'] = self.classes
        return config

    def init_from_config(self, config):
        super(ClassificationModel, self).init_from_config(config)
        self.classes = config['classes']

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
    def pred_class(self, preds):
        return ClassificationPredictionsResult(preds, names=self.classes)

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

def load_model_from_file(path_or_fp, net=None):
    return Model.load_from_file(path_or_fp, net)
