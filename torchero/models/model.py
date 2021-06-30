from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchero.utils.mixins import DeviceMixin
from torchero import meters
from torchero import SupervisedTrainer

class Model(DeviceMixin):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model
        self.trainer = None

    def compile(self, optimizer, criterion, metrics, hparams={}, callbacks=[], val_metrics=None):
        self.trainer = SupervisedTrainer(model=self.model,
                                         criterion=criterion,
                                         optimizer=optimizer,
                                         callbacks=callbacks,
                                         acc_meters=metrics,
                                         val_acc_meters=val_metrics,
                                         hparams=hparams)
        self.trainer.to(self.device)
        return self

    def predict_batch(self, *X):
        with torch.no_grad():
            X = map(self._convert_tensor, X)
            return self.model(*X)

    def to(self, device):
        super(Model, self).to(device)
        if self.trainer is not None:
            self.trainer.to(device)

    def predict_on_dataloader(self, dl):
        ys = []
        for X in dl:
            if isinstance(X, (tuple, list)):
                y = self.predict_batch(*X)
            else:
                y = self.predict_batch(X)
            ys.extend(y)
        return ys

    def predict(self, ds):
        dl = self._get_dataloader(train_ds,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  collate_fn=collate_fn,
                                  sampler=sampler)
        return self.predict_on_dataloader(train_dl)

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
                        sampler=None):
        if isinstance(ds, Dataset):
            dl = self._create_dataloader(ds,
                                         shuffle=shuffle,
                                         batch_size=batch_size or 32,
                                         collate_fn=collate_fn,
                                         sampler=sampler)
        elif isinstance(ds, DataLoader):
            dl = dl
        else:
            raise TypeError("ds type not supported. Use Dataloader or Dataset instances")
        return dl

    def evaluate(self,
                 ds,
                 metrics=None,
                 batch_size=None,
                 shuffle=True,
                 collate_fn=None,
                 sampler=None):
        dl = self._get_dataloader(ds,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
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

    def compile(self, optimizer, metrics=None, hparams={}, callbacks=[], val_metrics=None):
        criterion = 'binary_cross_entropy_wl' if self.use_logits else 'binary_cross_entropy'
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
                                                              criterion=criterion,
                                                              metrics=metrics,
                                                              hparams=hparams,
                                                              callbacks=callbacks,
                                                              val_metrics=val_metrics)

    def predict_batch(self, *X):
        preds = super(BinaryClassificationModel, self).predict_batch(*X)
        if self.use_logits:
            preds = preds > self.threshold
        return preds

class ClassificationModel(Model):
    def __init__(self, model, use_softmax=True):
        super(ClassificationModel, self).__init__(model)
        self.use_softmax = use_softmax

    def compile(self, optimizer, metrics=None, hparams={}, callbacks=[], val_metrics=None):
        criterion = 'cross_entropy' if self.use_softmax else 'nll'
        if metrics is None:
            metrics = [meters.CategoricalAccuracy(), meters.BalancedAccuracy()]
        return super(ClassificationModel, self).compile(optimizer=optimizer,
                                                        criterion=criterion,
                                                        metrics=metrics,
                                                        hparams=hparams,
                                                        callbacks=callbacks,
                                                        val_metrics=val_metrics)
    def predict_batch(self, *X):
        preds = super(BinaryClassificationModel, self).predict_batch(*X)
        if self.use_softmax:
            preds = nn.functional.softmax(preds)
        return preds

class RegressionModel(Model):
    def compile(self, optimizer, metrics=None, hparams={}, callbacks=[], val_metrics=None):
        criterion = 'mse'
        if metrics is None:
            metrics = [meters.RMSE()]
        return super(RegressionModel, self).compile(optimizer=optimizer,
                                                    criterion=criterion,
                                                    metrics=metrics,
                                                    hparams=hparams,
                                                    callbacks=callbacks,
                                                    val_metrics=val_metrics)
