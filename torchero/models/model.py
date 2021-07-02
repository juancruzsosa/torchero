import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torchero.utils.mixins import DeviceMixin
from torchero import meters
from torchero import SupervisedTrainer

class InputDataset(Dataset):
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.ds[idx])

    def __len__(self):
        return len(self.ds)

class Model(DeviceMixin):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model
        self.trainer = None

    def compile(self, optimizer, loss, metrics, hparams={}, callbacks=[], val_metrics=None):
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
        return X

    def _predict_batch(self, *X):
        self.model.train(False)
        with torch.no_grad():
            X = list(map(self._convert_tensor, X))
            return self.model(*X)

    def to(self, device):
        super(Model, self).to(device)
        if self.trainer is not None:
            self.trainer.to(device)

    def _combine_preds(self, preds):
        return torch.stack(preds).cpu()

    def predict_on_dataloader(self, dl, has_targets=True):
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
        dl = self._get_dataloader(ds,
                                  shuffle=False,
                                  batch_size=batch_size,
                                  shallow_dl=to_tensor)
        return self.predict_on_dataloader(dl, has_targets=has_targets)

    def train_on_dataloader(self, train_dl, val_dl=None, epochs=1):
        if self.trainer is None:
            raise Exception("Model hasn't been compiled with any trainer. Use model.compile first")

        self.trainer.train(dataloader=train_dl,
                           valid_dataloader=val_dl,
                           epochs=epochs)
        return self.trainer.history

    def evaluate_on_dataloader(self,
                               dataloader,
                               metrics=None):
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

class BinaryClassificationModel(Model):
    def __init__(self, model, use_logits=True, threshold=0.5):
        super(BinaryClassificationModel, self).__init__(model)
        self.use_logits = use_logits
        self.threshold = threshold

    def compile(self, optimizer, loss=None, metrics=None, hparams={}, callbacks=[], val_metrics=None):
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
    def __init__(self, model, use_softmax=True):
        super(ClassificationModel, self).__init__(model)
        self.use_softmax = use_softmax

    def compile(self, optimizer, loss=None, metrics=None, hparams={}, callbacks=[], val_metrics=None):
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
    def compile(self, optimizer, loss=None, metrics=None, hparams={}, callbacks=[], val_metrics=None):
        if loss is None:
            loss = 'mse'
        if metrics is None:
            metrics = [meters.RMSE()]
        return super(RegressionModel, self).compile(optimizer=optimizer,
                                                    loss=loss,
                                                    metrics=metrics,
                                                    hparams=hparams,
                                                    callbacks=callbacks,
                                                    val_metrics=val_metrics)
