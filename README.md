# Torchtrainer - A pluggable & extensible trainer for pytorch #

## Features ##

* Train/validate models for given number of epochs
* Hooks/Callbacks to add personalized behavior
* Different metrics of model accuracy/error
* Training/validation statistics monitors
* Cross fold validation iterators for splitting validation data from train data

### Trainers ###

* `BatchTrainer`: Abstract class for all trainers that works with batched inputs
* `SupervisedTrainer`: Supervised trainer

### Hooks ###

* `hooks.Hook`: Base hook class for all epoch/training events
* `hooks.History`: Hook that record history of all training/validation metrics
* `hooks.ProgressBars`: Hook that displays progress bars to monitor training/validation metrics
* `hooks.Container`: Hook to group multiple hooks
* `hooks.CSVExporter`: Hook that export training/validation stadistics to a csv file

### Metrics ###

* `metrics.Base`: Interface for all meters
* `metrics.CategoricalAccuracy`: Meter for accuracy on categorical targets
* `metrics.BinaryAccuracy`: Meter for accuracy on binary targets (assuming normalized inputs)
* `metrics.BinaryAccuracyWithLogits`: Binary accuracy meter with an integrated activation function (by default logistic function)
* `metrics.MSE`: Mean Squared Error meter

### Cross validation ###

* `utils.data.CrossFoldValidation`: Itererator through cross-fold-validation folds
