# Torchero - A training framework for pytorch #

[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/juancruzsosa/torchero/Python%20package?logo=github)](https://github.com/juancruzsosa/torchero/actions)
[![codecov](https://codecov.io/gh/juancruzsosa/torchero/branch/master/graph/badge.svg)](https://codecov.io/gh/juancruzsosa/torchero)
[![PyPI](https://img.shields.io/pypi/v/torchero?logo=pypi)](https://pypi.org/project/torchero/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchero?label=python&logo=python)](https://www.python.org/downloads/)
[![license: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features ##

* Train/validate models for given number of epochs
* Hooks/Callbacks to add personalized behavior
* Different metrics of model accuracy/error
* Training/validation statistics monitors
* Cross fold validation iterators for splitting validation data from train data

## Installation ##

### From PyPI ###

```bash
pip install torchero
```

### From Source Code ###

```bash
git clone https://github.com/juancruzsosa/torchero
cd torchero
python setup.py install
```

## Example ##

### Training with MNIST 

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
import torchero
from torchero import SupervisedTrainer
from torchero.meters import CategoricalAccuracy
from torchero.callbacks import ProgbarLogger as Logger, CSVLogger

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.filter = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm2d(32),
                                    nn.MaxPool2d(2),
                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm2d(64),
                                    nn.MaxPool2d(2))
        self.linear = nn.Sequential(nn.Linear(5*5*64, 500),
                                    nn.BatchNorm1d(500),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(500, 10))

    def forward(self, x):
        bs = x.shape[0]
        return self.linear(self.filter(x).view(bs, -1))

train_ds = MNIST(root='data/',
                 download=True,
                 train=True,
                 transform=transforms.Compose([transforms.ToTensor()]))
test_ds = MNIST(root='data/',
                download=False,
                train=False,
                transform=transforms.Compose([transforms.ToTensor()]))
train_dl = DataLoader(train_ds, batch_size=50)
test_dl = DataLoader(test_ds, batch_size=50)

model = Network()

trainer = SupervisedTrainer(model=model,
                            optimizer='sgd',
                            criterion='cross_entropy',
                            acc_meters={'acc': 'categorical_accuracy_percentage'},
                            callbacks=[Logger(),
                                       CSVLogger(output='training_stats.csv')
                                      ])

# If you want to use cuda uncomment the next line
# trainer.cuda()

trainer.train(dataloader=train_dl,
              valid_dataloader=test_dl,
              epochs=2)

```

### Trainers ###

* `BatchTrainer`: Abstract class for all trainers that works with batched inputs
* `SupervisedTrainer`: Training for supervised tasks
* `AutoencoderTrainer`: Trainer for auto encoder tasks

### Callbacks ###

* `callbacks.Callback`: Base callback class for all epoch/training events
* `callbacks.History`: Callback that record history of all training/validation metrics
* `callbacks.Logger`: Callback that display metrics per logging step
* `callbacks.ProgbarLogger`: Callback that displays progress bars to monitor training/validation metrics
* `callbacks.CallbackContainer`: Callback to group multiple hooks
* `callbacks.ModelCheckpoint`: Callback to save best model after every epoch
* `callbacks.EarlyStopping`: Callback to stop training when monitored quanity not improves
* `callbacks.CSVLogger`: Callback that export training/validation stadistics to a csv file

### Meters ###

* `meters.BaseMeter`: Interface for all meters
* `meters.BatchMeters`: Superclass of meters that works with batchs
* `meters.CategoricalAccuracy`: Meter for accuracy on categorical targets
* `meters.BinaryAccuracy`: Meter for accuracy on binary targets (assuming normalized inputs)
* `meters.BinaryAccuracyWithLogits`: Binary accuracy meter with an integrated activation function (by default logistic function)
* `meters.ConfusionMatrix`: Meter for confusion matrix.
* `meters.MSE`: Mean Squared Error meter
* `meters.MSLE`: Mean Squared Log Error meter
* `meters.RMSE`: Rooted Mean Squared Error meter
* `meters.RMSLE`: Rooted Mean Squared Log Error meter
* `meters.Precision`: Precision meter
* `meters.Recall`: Precision meter
* `meters.Specificity`: Precision meter
* `meters.NPV`:  Negative predictive value meter
* `meters.F1Score`: F1 Score meter
* `meters.F2Score`: F2 Score meter

### Cross validation ###

* `utils.data.CrossFoldValidation`: Itererator through cross-fold-validation folds
* `utils.data.train_test_split`: Split dataset into train and test datasets

### Datasets ###

* `utils.data.datasets.SubsetDataset`: Dataset that is a subset of the original dataset
* `utils.data.datasets.ShrinkDatset`: Shrinks a dataset
* `utils.data.datasets.UnsuperviseDataset`: Makes a dataset unsupervised
