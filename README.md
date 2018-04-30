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

### Callbacks ###

* `callback.Callback`: Base callback class for all epoch/training events
* `callback.History`: Callback that record history of all training/validation metrics
* `callback.ProgressBars`: Callback that displays progress bars to monitor training/validation metrics
* `callback.CallbackContainer`: Callback to group multiple hooks
* `callback.CSVLogger`: Callback that export training/validation stadistics to a csv file

### Meters ###

* `meter.Base`: Interface for all meters
* `meter.CategoricalAccuracy`: Meter for accuracy on categorical targets
* `meter.BinaryAccuracy`: Meter for accuracy on binary targets (assuming normalized inputs)
* `meter.BinaryAccuracyWithLogits`: Binary accuracy meter with an integrated activation function (by default logistic function)
* `meter.MSE`: Mean Squared Error meter

### Cross validation ###

* `utils.data.CrossFoldValidation`: Itererator through cross-fold-validation folds
