# Torchero - A training framework for pytorch #

**Torchero** is a library that works on top of the *PyTorch* framework built to
facilitate the training of Neural Networks.

[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/juancruzsosa/torchero/Python%20package?logo=github)](https://github.com/juancruzsosa/torchero/actions)
[![codecov](https://codecov.io/gh/juancruzsosa/torchero/branch/master/graph/badge.svg)](https://codecov.io/gh/juancruzsosa/torchero)
[![PyPI](https://img.shields.io/pypi/v/torchero?logo=pypi)](https://pypi.org/project/torchero/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchero?label=python&logo=python)](https://www.python.org/downloads/)
[![license: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/torchero/badge/?version=latest)](https://torchero.readthedocs.io/en/latest/?badge=latest)

## Features ##

It provides tools and utilities to:

- Set up a training process in a few lines of code.
- Monitor the training performance by checking several prebuilt metrics on a handy progress bar.
- Integrate a dashboard with *TensorBoard* to visualize those metrics in an online manner with a minimal setup.
- Add custom functionality via Callbacks.
- NLP: Datasets for text classification tasks. Vectors, etc.

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

## Quickstart - MNIST ##

In this case we are going to use torchero to train a Convolutional Neural Network
for the MNIST Dataset.

### Loading the Data

First, we load the dataset using ``torchvision``. Then,
if we want we can show the image samples using ``show_imagegrid_dataset``

```python
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST # The MNIST dataset
from torchvision import transforms # To convert to Images to Tensors

from torchero import SupervisedTrainer
from torchero.callbacks import ProgbarLogger, ModelCheckpoint, CSVLogger
from torchero.utils import show_imagegrid_dataset

from matplotlib import pyplot as plt

train_ds = MNIST(root='/tmp/data/mnist', download=True, train=True, transform=transforms.ToTensor())
test_ds = MNIST(root='/tmp/data/mnist', download=False, train=False, transform=transforms.ToTensor())

train_dl = DataLoader(train_ds, batch_size=32)
test_dl = DataLoader(test_ds, batch_size=32)

show_imagegrid_dataset(train_ds)
plt.show()
```

![mnist images by class](documentation/source/img/quickstart/mnist_train_data.png)

### Creating the Model

Then we have to define the model. For this case we can use a Sequential one.

```python
model = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
					  nn.ReLU(inplace=True),
					  nn.MaxPool2d(2),
					  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
					  nn.ReLU(inplace=True),
					  nn.MaxPool2d(2),
					  nn.Flatten(),
					  nn.Linear(5*5*64, 500),
					  nn.ReLU(inplace=True),
					  nn.Linear(500, 10))
```

### Training the Model

After the network and both train and test DataLoader's are defined.
We can run the training using ``SDG`` optimizer, cross entropy Loss,
categorical accuracy, a progress bar, a ModelCheckpoint to
save the model when improves on accuracy, a CSVLogger to
dump the metrics on a CSV file. We can call ``trainer.cuda()`` if we
want to do training on GPU insted of CPU.

```python
trainer = SupervisedTrainer(model=model,
						  optimizer='sgd',
						  criterion='cross_entropy',
						  acc_meters=['categorical_accuracy_percentage'],
						  callbacks=[ProgbarLogger(notebook=True),
									 ModelCheckpoint('saved_model', mode='max', monitor='val_acc'),
									 CSVLogger('training_results.xml')])

if torch.cuda.is_available():
	trainer.cuda()

trainer.train()
trainer.train(dataloader=train_dl, valid_dataloader=test_dl, epochs=5)
```

![progress bar training](documentation/source/img/quickstart/training_status.gif)

### Showing the training results

To see the training metrics

```python
fig, axs = plt.subplots(figsize=(14,3), ncols=2, nrows=1)
trainer.history.plot()
plt.show()
```

And if we want to see for example the confusion matrix on the test set.

```python
results = trainer.evaluate(test_dl, ['categorical_accuracy_percentage', 'confusion_matrix'])
plt.figure(figsize=(10, 10))
results['confusion_matrix'].plot(classes=train_ds.classes)
```
![confusion matrix](documentation/source/img/quickstart/confusion_matrix.png)

## Documentation ##

Additional documentation can be founded [here](https://readthedocs.org/projects/torchero/badge/?version=latest)

## Extensive List of Classes ##

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
