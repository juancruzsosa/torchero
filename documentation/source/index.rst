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

**Torchero** is a library that works on top of the *PyTorch* framework built to
facilitate the training of Neural Networks.

Features
--------

It provides tools and utilities to:

- Set up a training process in a few lines of code.
- Monitor the training performance by checking several prebuilt metrics on a handy progress bar.
- Integrate a dashboard with *TensorBoard* to visualize those metrics in an online manner with a minimal setup.
- Add custom functionality via Callbacks.

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

.. code-block:: python

    import torch
    from torch import nn

    import torchero
    from torchero.models.vision import ImageClassificationModel
    from torchero.callbacks import ProgbarLogger, ModelCheckpoint, CSVLogger
    from torchero.utils.data import train_test_split
    from torchero.utils.vision import show_imagegrid_dataset, transforms, datasets, download_image
    from torchero.meters import ConfusionMatrix

    from matplotlib import pyplot as plt


|   First we load the MNIST train and test datasets and visualize it using ``show_imagegrid_dataset``.
|   The Data Augmentation for this case will be a RandomInvert to flip the grayscale levels.

.. code-block:: python

    train_ds = datasets.MNIST(root='/tmp/data/mnist', download=True, train=True, transform=transforms.Compose([transforms.RandomInvert(),
    test_ds = datasets.MNIST(root='/tmp/data/mnist', download=False, train=False, transform=transforms.ToTensor())
    show_imagegrid_dataset(train_ds)
    plt.show()

.. image:: /img/quickstart/install_mnist.png
.. image:: /img/quickstart/mnist_train_data.png

Defining the Network
--------------------

| Let's define a Convolutional network of two layers followed by a Linear Module
| as the classification layer.

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

Training the Model
------------------

|  The ImageClassificationModel is the module responsible to train the model,
|  evaluate a metric against a dataset, and predict from and input for multi-class classification tasks.
|
|  To train the model we need to compile it first with a:
|
|  - an optimizer: 'adam'
|  - a loss which will be defaulted to ``cross_entropy``
|  - a list of metrics which will be defaulted to ``categorical_accuracy``, ``balanced_accuracy``)
|  - a list of callbacks:
|      - ProgbarLogger to show training progress bar
|      - ModelCheckpoint to make checkpoints if the model improves
|      - CSVLogger to dump the metrics to a csv file after each epoch

.. code-block:: python

    model = ImageClassificationModel(model=network, 
                                     transform=transforms.Compose([transforms.Grayscale(),
                                                                   transforms.Resize((28,28)),
                                                                   transforms.ToTensor()]),
                                     classes=[str(i) for i in range(10)])
    model.compile(optimizer='adam',
                  callbacks=[ProgbarLogger(notebook=True),
                             ModelCheckpoint('saved_model', mode='max', monitor='val_acc'),
                             CSVLogger('training_results.xml')])

    if torch.cuda.is_available():
        model.cuda()

    history = model.fit(train_ds,
                        test_ds,
                        batch_size=1024,
                        epochs=5)

.. image:: /img/quickstart/training_status.gif

Visualizing the training training results
-----------------------------------------

To visualize our loss and accuracy in each epoch we
can execute:

.. code-block:: python

    history.plot(figsize=(20, 20), smooth=0.2)
    plt.show()

.. image:: /img/quickstart/metrics.png

The ``.evaluate`` returns the metrics for a new dataset.

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
