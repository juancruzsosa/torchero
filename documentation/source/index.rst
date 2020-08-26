.. highlight:: bash
.. torchero documentation master file, created by
   sphinx-quickstart on Tue Aug 25 18:26:22 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to torchero's documentation!
====================================

.. image:: https://img.shields.io/github/workflow/status/juancruzsosa/torchero/Python%20package?logo=github
    :target: https://github.com/juancruzsosa/torchero/actions
    :alt: GitHub Workflow Status

.. image:: https://codecov.io/gh/juancruzsosa/torchero/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/juancruzsosa/torchero
    :alt: Code Coverage

.. image:: https://img.shields.io/pypi/v/torchero?logo=pypi
    :target: https://pypi.org/project/torchero/
    :alt: PyPI

.. image:: https://img.shields.io/pypi/pyversions/torchero?label=python&logo=python
    :target: https://www.python.org/downloads/
    :alt: Python Version: 3

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License: MIT

.. image:: https://readthedocs.org/projects/torchero/badge/?version=latest
    :target: https://torchero.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

**Torchero** is a library that works on top of *PyTorch* framework intended to
facilitate training of Neural Networks.

Features
--------

It provides tools and utilities to

- Setup a training process in few lines of code.
- Comes prebuilt with several of metrics to monitor the training performance and display them in a handy progress bar.
- If it's required, it integrates with *TensorBoard* to visualize those metrics in an online manner with a minimal setup.
- Adaptability to add functionality vía Callbacks.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   Installation

Installlation
=============

You can install ``torchero`` either from pip or from source.

From pip
--------

.. code-block:: bash

    python3 -m pip install torchero

From source code
----------------

Download the source code using git

.. code-block:: bash

    git clone https://github.com/juancruzsosa/torchero

or download the tarball if you don't have *git* installed

.. code-block:: bash

   curl -OL https://github.com/juancruzsosa/torchero/tarball/master

Once you have the source code downloaded you can install it with

.. code-block:: bash

   cd torchero
   python3 setup.py install


Quickstart
==========

Loading the Data
----------------

First we need to import torchvision libraries to load the dataset

.. code-block:: python

    from torchvision.datasets import MNIST # The MNIST dataset
    from torchvision import transforms # To convert to Images to Tensors


Now we can import the train and the test dataset

.. code-block:: python

    train_ds = MNIST(root='/tmp/data/mnist', download=True, train=True, transform=transforms.ToTensor())
    test_ds = MNIST(root='/tmp/data/mnist', download=False, train=False, transform=transforms.ToTensor())

.. image:: /img/quickstart/install_mnist.png

| To visualize the train dataset we can use show_imagegrid_dataset from utils
| subpackage. We can do the same for the test set. This could be helpfull to see
| if just with a quick look we can see if the test dataset is representative with
| respect to the train dataset.

.. code-block:: python

    from matplotlib import pyplot as plt
    from torchero.utils import show_imagegrid_dataset

    show_imagegrid_dataset(train_ds)
    plt.show()

.. image:: /img/quickstart/mnist_train_data.png

| Then we need to setup one DataLoader for train dataset and another one for
| test dataset. This component is responsible to fetch the training and test
| data in form of batches. For the batch size parameter we can use 32.

.. code-block:: python

    train_dl = DataLoader(train_ds, batch_size=32)
    test_dl = DataLoader(test_ds, batch_size=32)


Model definition
----------------

| Then we need to define the architecture of the model. For this case we can
| define a simple Sequential network with 2 Convolutional Layers and 2 dense ones
| as follows:

.. code-block:: python

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

Training
--------

The SupervisedTrainer is the responsible to train our network. We need to construct it with at least 3 things

- The Model instance (we have already do that)
- The Optimizer. Here we can choose ``'adam'`` (there are other multiple choices like ``'sdg'``, ``'asgd'``, ``'adagrad'``, ``'adamax'``, ``'lbfgfs'``, etc.)
- The Criterion (or loss). Given the fact that this is a classification problem we can use ``'cross_entropy'``

Some additional things we can pass it to the trainer
are

- A list or dictionary of metrics. For this example we are going to choose ``'categorical_accuracy_percentage'``
- A list of callbacks. For this list we will choose

  - ``ProgbarLogger``: To report a Progress Bar to see the status while Training. It's important to note here that if you are running the code over a Jupyter notebook you may want to set `notebook` parameter to true to display the progress bar in HTML format.
  - ``ModelCheckpoint``: To save our model if this one improves in `categorical_accuracy_percentage`
  - ``CSVLogger``: To save the table of metrics to be able to share it or load it from other program if we want

.. code-block:: python

  from torchero import SupervisedTrainer
  from torchero.callbacks import ProgbarLogger, ModelCheckpoint, CSVLogger

  trainer = SupervisedTrainer(model=model,
                              optimizer='sgd',
                              criterion='cross_entropy',
                              acc_meters=['categorical_accuracy_percentage'],
                              callbacks=[ProgbarLogger(notebook=True),
                                         ModelCheckpoint('saved_model', mode='max', monitor='val_acc'),
                                         CSVLogger('training_results.xml')])

If we want to train using GPU. We can just execute

.. code-block:: python

  trainer.cuda()

This will automatically convert the model and the data from which the model will feed to cuda as well.

Finally we need to train our network. We can
do it we this simple line.

.. code-block:: python

  trainer.train(dataloader=train_dl, valid_dataloader=test_dl, epochs=5)

.. image:: /img/quickstart/training_status.gif

Visualizing the training training results
-----------------------------------------

To visualize our loss and accuracy in each epoch we
can execute:

.. code-block:: python

  fig, axs = plt.subplots(figsize=(14,3), ncols=2, nrows=1)
  trainer.history.epoch_plot(['train_acc', 'val_acc'], ax=axs[0], title="Accuracy")
  trainer.history.epoch_plot(['train_loss', 'val_loss'], ax=axs[1], title="Loss")
  plt.show()

.. image:: /img/quickstart/metrics.png

Also, if we want to see how well the model
performed on each label we can show a confusion matrix as following.

.. code-block:: python

  results = trainer.evaluate(test_dl, ['categorical_accuracy_percentage', 'confusion_matrix'])
  plt.figure(figsize=(10, 10))
  results['confusion_matrix'].plot(classes=train_ds.classes)

.. image:: /img/quickstart/confusion_matrix.png


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
