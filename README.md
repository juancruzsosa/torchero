# Torchtrainer - A training framework for pytorch #

## Features ##

* Train/validate models for given number of epochs
* Hooks/Callbacks to add personalized behavior
* Different metrics of model accuracy/error
* Training/validation statistics monitors
* Cross fold validation iterators for splitting validation data from train data

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

### Cross validation ###

* `utils.data.CrossFoldValidation`: Itererator through cross-fold-validation folds

### Datasets ###

* `utils.data.datasets.SubsetDataset`: Dataset that is a subset of the original dataset
* `utils.data.datasets.ShrinkDatset`: Shrinks a dataset
* `utils.data.datasets.UnsuperviseDataset`: Makes a dataset unsupervised
