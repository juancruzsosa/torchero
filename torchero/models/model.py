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

class ModelNotCompiled(Exception):
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

    def __repr__(self):
        list_format = []
        for pred in self._preds[:10]:
            list_format.append(repr(pred))
        if len(self._preds) > 10:
            list_format.append('...')
        format_string = '{}([{}])'.format(self.__class__.__name__,
                                          '\n,'.join(list_format))
        return format_string

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
        self._trainer = None

    def pred_class(self, preds):
        return PredictionsResult(preds)

    @property
    def trainer(self):
        if self._trainer is None:
            raise ModelNotCompiled("Model hasn't been compiled with any trainer. Use model.compile first")
        return self._trainer

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
        self._trainer = SupervisedTrainer(model=self.model,
                                         criterion=loss,
                                         optimizer=optimizer,
                                         callbacks=callbacks,
                                         acc_meters=metrics,
                                         val_acc_meters=val_metrics,
                                         hparams=hparams)
        self._trainer.to(self.device)
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

    @property
    def callbacks(self):
        return self.trainer.callbacks

    @property
    def optimizer(self):
        return self.trainer.optimizer

    @optimizer.setter
    def optimizer(Self, optimizer):
        self.trainer.optimizer = optimizer

    @property
    def hparams(self):
        return dict(self.trainer.hparams)

    @property
    def history(self):
        return self.trainer.history

    @property
    def loss(self):
        return self.trainer.criterion

    @loss.setter
    def loss(self, loss):
        self.trainer.criterion = loss

    def to(self, device):
        """ Moves the model to the given device

        Arguments:
            device (str or torch.device)
        """
        super(Model, self).to(device)
        try:
            self.trainer.to(device)
        except ModelNotCompiled:
            pass

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
                X, _ = X
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
                has_targets=False,
                num_workers=0,
                pin_memory=False,
                prefetch_factor=2):
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
            num_workers (int, optional): Number of subprocesses to use for data
                loading. 0 means that the data will be loaded in the main process.
            pin_memory (bool): If True, the data loader will copy Tensors into
                CUDA pinned memory before returning them. If your data elements are
                a custom type, or your collate_fn returns a batch that is a custom
                type, see the example below.
            prefetch_factor (int, optional):
                Number of samples loaded in advance by each worker. 2 means
                there will be a total of 2 * num_workers samples prefetched
                across all workers.
        """
        dl = self._get_dataloader(ds,
                                  shuffle=False,
                                  batch_size=batch_size,
                                  shallow_dl=to_tensor,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  prefetch_factor=prefetch_factor)
        return self.predict_on_dataloader(dl, has_targets=has_targets)

    def train_on_dataloader(self, train_dl, val_dl=None, epochs=1):
        """ Trains the model for a fixed number of epochs

        Arguments:
            train_ds (`torch.utils.data.DataLoader`): Train dataloader
            val_ds (`torch.utils.data.Dataset`): Test dataloader
            epochs (int): Number of epochs to train the model
        """
        self.trainer.train(dataloader=train_dl,
                           valid_dataloader=val_dl,
                           epochs=epochs)
        return self.trainer.history

    def load_checkpoint(self, checkpoint=None):
        self.trainer.load_checkpoint(checkpoint=None)

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
                        shallow_dl=False,
                        **dl_kwargs):
        if isinstance(ds, (Dataset, list)):
            dl = self._create_dataloader(InputDataset(ds, self.input_to_tensor) if shallow_dl else ds,
                                         batch_size=batch_size or 32,
                                         **dl_kwargs)
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
                 sampler=None,
                 num_workers=0,
                 pin_memory=False,
                 prefetch_factor=2):
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
            num_workers (int, optional): Number of subprocesses to use for data
                loading. 0 means that the data will be loaded in the main process.
            pin_memory (bool): If True, the data loader will copy Tensors into
                CUDA pinned memory before returning them. If your data elements are
                a custom type, or your collate_fn returns a batch that is a custom
                type, see the example below.
            prefetch_factor (int, optional): 
                Number of samples loaded in advance by each worker. 2 means
                there will be a total of 2 * num_workers samples prefetched
                across all workers.
        """
        dl = self._get_dataloader(ds,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  collate_fn=collate_fn,
                                  sampler=sampler,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  prefetch_factor=prefetch_factor)
        return self.evaluate_on_dataloader(dl, metrics=metrics)

    def fit(self,
            train_ds,
            val_ds=None,
            epochs=1,
            batch_size=None,
            shuffle=True,
            collate_fn=None,
            sampler=None,
            num_workers=0,
            val_num_workers=None,
            pin_memory=False,
            val_pin_memory=False,
            prefetch_factor=2,
            val_prefetch_factor=None):
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
            num_workers (int, optional): Number of subprocesses to use for data
                loading. 0 means that the data will be loaded in the main process.
            val_num_workers (int, optional): Same as num_workers but for the validation dataset.
                If not passed num_workers argument will be used
            pin_memory (bool): If True, the data loader will copy Tensors into
                CUDA pinned memory before returning them. If your data elements are
                a custom type, or your collate_fn returns a batch that is a custom
                type, see the example below.
            val_pin_memory (bool): Same as pin_memory but for the validation dataset.
                If not passed pin_memory argument will be used
            prefetch_factor (int, optional): 
                Number of samples loaded in advance by each worker. 2 means
                there will be a total of 2 * num_workers samples prefetched
                across all workers.
            val_prefetch_factor (int, optional): Same as prefetch_factor but for the validation dataset.
                If not passed prefetch_factor argument will be used
        """
        train_dl = self._get_dataloader(train_ds,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       collate_fn=collate_fn,
                                       sampler=sampler,
                                       num_workers=num_workers,
                                       pin_memory=pin_memory,
                                       prefetch_factor=prefetch_factor)
        if val_ds is None:
            val_dl = None
        else:
            val_dl = self._get_dataloader(val_ds,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          collate_fn=collate_fn,
                                          sampler=sampler,
                                          num_workers=val_num_workers or num_workers,
                                          pin_memory=val_pin_memory or pin_memory,
                                          prefetch_factor=val_prefetch_factor or prefetch_factor)
        return self.train_on_dataloader(train_dl,
                                        val_dl,
                                        epochs)

    @property
    def config(self):
        config = {
            'torchero_version': torchero.__version__,
            'torchero_model_type': {'module': self.__class__.__module__,
                                    'type': self.__class__.__name__},
            'compiled': self._trainer is not None,
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
        try:
            self.trainer._save_to_zip(zip_fp, prefix='trainer/')
        except ModelNotCompiled:
            pass

    def load(self, path_or_fp):
        with zipfile.ZipFile(path_or_fp, mode='r') as zip_fp:
            self._load_from_zip(zip_fp)

    def _load_from_zip(self, zip_fp):
        with zip_fp.open('model.pth', 'r') as fp:
            self.model.load_state_dict(torch.load(fp))
        with zip_fp.open('config.json', 'r') as config_fp:
            config = json.loads(config_fp.read().decode())
        if config['compiled'] is True:
            self._trainer = SupervisedTrainer(model=self.model,
                                             criterion=None,
                                             optimizer=None)
            self._trainer._load_from_zip(zip_fp, prefix='trainer/')
        self.init_from_config(config)

class UnamedClassificationPredictionItem(PredictionItem):
    """ Model Prediction with classes names
    """
    def __init__(self, preds):
        super(UnamedClassificationPredictionItem, self).__init__(preds)
        if self._preds.ndim == 0:
            self._preds = self._preds.unsqueeze(-1)

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

    def __repr__(self):
        return repr(self.as_tuple())

class NamedClassificationPredictionItem(PredictionItem):
    """ Model Prediction with classes names
    """
    def __init__(self, preds, names=None):
        super(NamedClassificationPredictionItem, self).__init__(preds)
        self.names = names
        if self._preds.ndim == 0:
            self._preds = self._preds.unsqueeze(-1)

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

    def __repr__(self):
        return repr(self.as_dict())

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

    def classification_report(self,
                              ds,
                              batch_size=None,
                              collate_fn=None,
                              sampler=None):
        clf_report = meters.binary_scores.BinaryClassificationReport(threshold=self.threshold,
                                                                     with_logits=self.use_logits,
                                                                     names=self.labels)
        metrics = self.evaluate(ds,
                                metrics={'clf': clf_report},
                                batch_size=batch_size,
                                collate_fn=collate_fn,
                                sampler=sampler)
        return metrics['clf']

    def _predict_batch(self, *X, output_probas=True):
        preds = super(BinaryClassificationModel, self)._predict_batch(*X)
        if self.use_logits:
            preds = torch.sigmoid(preds)
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
            preds = torch.softmax(preds, dim=-1)
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
