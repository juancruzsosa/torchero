import shutil
import pandas as pd

import torchero

from .common import *

class HistoryCallbackTests(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        train_dataset = torch.arange(10).view(-1, 1)
        valid_dataset = torch.arange(10).view(-1, 1)

        self.train_dl = DataLoader(train_dataset, shuffle=False, batch_size=1)
        self.valid_dl = DataLoader(valid_dataset, shuffle=False, batch_size=1)

        def update_batch(trainer, x):
            trainer.train_meters['t_c'].measure(x.data[0][0])

        def validate_batch(validator, x):
            validator.meters['v_c'].measure(x.data[0][0])

        self.trainer = TestTrainer(model=self.model,
                                   logging_frecuency=5,
                                   train_meters={'t_c' : Averager()},
                                   val_meters={'v_c' : Averager()},
                                   update_batch_fn=update_batch,
                                   valid_batch_fn=validate_batch)

    def test_repr(self):
        self.assertEqual(repr(History()), "History()")

    def test_history_callback_register_every_training_stat(self):
        self.assertEqual(self.trainer.metrics, {})

        self.assertEqual(set(self.trainer.meters_names()), set(['t_c', 'v_c']))

        self.trainer.train(self.train_dl, valid_dataloader=self.valid_dl, epochs=1)

        expected_registry = [{'epoch': 0, 'step': 5, 't_c': 2.0, 'v_c': 4.5},
                             {'epoch': 0, 'step': 10, 't_c': 4.5, 'v_c': 4.5}]

        self.assertEqual(list(self.trainer.history), expected_registry)
        self.assertEqual(self.trainer.metrics, {'t_c': 4.5, 'v_c': 4.5})

    def test_export_to_dataframe(self):
        self.trainer.train(self.train_dl, valid_dataloader=self.valid_dl, epochs=2)

        try:
            self.trainer.history.to_dataframe(level='xxx')
            self.fail()
        except Exception as e:
            self.assertIsInstance(e, ValueError)
            self.assertEqual(str(e), self.trainer.history.UNRECOGNIZED_LEVEL_MESSAGE.format(level=repr('xxx')))

        df_history = self.trainer.history.to_dataframe(level='step')
        self.assertIsInstance(df_history, pd.DataFrame)
        df_expected = pd.DataFrame({'epoch': [0, 0, 1, 1],
                                    'step':  [5, 10, 15, 20],
                                    't_c':   [2.0, 4.5, 2.0, 4.5],
                                    'v_c':   [4.5, 4.5, 4.5, 4.5]})
        pd.testing.assert_frame_equal(df_history, df_expected)

        df_history = self.trainer.history.to_dataframe(level='epoch')
        df_expected = pd.DataFrame({'epoch': [0, 1],
                                    't_c':   [4.5, 4.5],
                                    'v_c':   [4.5, 4.5]})
        pd.testing.assert_frame_equal(df_history, df_expected)


class CSVExporterTests(unittest.TestCase):
    def setUp(self):
        self.base_tree = os.path.join('tests', 'output')
        self.load_model()
        os.makedirs(self.base_tree)

        self.stats_filename = os.path.join(self.base_tree, 'stats.csv')
        self.stats_filename_epoch = os.path.join(self.base_tree, 'stats_epoch.csv')

    def load_model(self):
        self.model = DummyModel()

    def load_empty_dataset(self):
        self.train_ds = torch.Tensor([])
        self.train_dl = DataLoader(self.train_ds, shuffle=False, batch_size=1)

    def load_arange_dataset(self, limit=10, batch_size=1):
        self.train_ds = torch.arange(limit).view(-1, 1)
        self.train_dl = DataLoader(self.train_ds, shuffle=False, batch_size=batch_size)

    @staticmethod
    def update_batch(trainer, x):
        trainer.train_meters['c'].measure(x.data[0][0])

    def csv_logger(self, append=False, columns=None):
        return CSVLogger(output=self.stats_filename, append=append, columns=columns, level='step')

    def csv_epoch_logger(self, append=False, columns=None):
        return CSVLogger(output=self.stats_filename_epoch, append=append, columns=columns)

    def test_unknown_level_parameter_raises_error(self):
        try:
            logger = CSVLogger(self.stats_filename, level='xxx')
            self.fail()
        except ValueError as e:
            self.assertEqual(str(e), CSVLogger.UNRECOGNIZED_LEVEL.format(level=repr('xxx')))
            self.assertFalse(os.path.exists(self.stats_filename))

    def test_csv_exporter_print_header_at_begining_of_training(self):
        self.load_empty_dataset()

        callback_1 = self.csv_logger(append=False)
        callback_2 = self.csv_epoch_logger(append=False)

        self.assertFalse(os.path.exists(self.stats_filename))
        self.assertFalse(os.path.exists(self.stats_filename_epoch))

        trainer = TestTrainer(model=self.model,
                              callbacks=[callback_1, callback_2],
                              train_meters={'c' : Averager()},
                              logging_frecuency=5,
                              update_batch_fn=self.update_batch)

        trainer.train(self.train_dl, epochs=1)

        with open(self.stats_filename, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0], 'epoch,step,c')

        with open(self.stats_filename_epoch, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0], 'epoch,c')

    def test_csv_exporter_stats_write_stats_to_csv_after_every_log(self):
        self.load_arange_dataset()

        callback_1 = self.csv_logger(append=False)
        callback_2 = self.csv_epoch_logger(append=False)
        self.assertFalse(os.path.exists(self.stats_filename))
        self.assertFalse(os.path.exists(self.stats_filename_epoch))

        trainer = TestTrainer(model=self.model,
                              logging_frecuency=5,
                              train_meters={'c': Averager()},
                              update_batch_fn=self.update_batch)
        trainer.add_callback(callback_1)
        trainer.add_callback(callback_2)
        trainer.train(self.train_dl, epochs=1)

        with open(self.stats_filename, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0], 'epoch,step,c\n')
            self.assertEqual(lines[1], '0,5,2.0\n')
            self.assertEqual(lines[2], '0,10,4.5')

        with open(self.stats_filename_epoch, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0], 'epoch,c\n')
            self.assertEqual(lines[1], '1,4.5')



    def test_export_stats_can_append_stats_with_matching_cols_to_previous_training(self):
        self.load_arange_dataset()

        callback_1 = self.csv_logger(append=True)
        callback_2 = self.csv_epoch_logger(append=True)
        self.assertFalse(os.path.exists(self.stats_filename))

        trainer = TestTrainer(model=self.model,
                              callbacks=[callback_1, callback_2],
                              logging_frecuency=5,
                              train_meters={'c': Averager()},
                              update_batch_fn=self.update_batch)

        trainer.train(self.train_dl, epochs=1)
        trainer.train(self.train_dl, epochs=1)

        with open(self.stats_filename, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 5)
            self.assertEqual(lines[0], 'epoch,step,c\n')
            self.assertEqual(lines[1], '0,5,2.0\n')
            self.assertEqual(lines[2], '0,10,4.5\n')
            self.assertEqual(lines[3], '1,15,2.0\n')
            self.assertEqual(lines[4], '1,20,4.5')

        with open(self.stats_filename_epoch, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0], 'epoch,c\n')
            self.assertEqual(lines[1], '1,4.5\n')
            self.assertEqual(lines[2], '2,4.5')

    def test_csv_exporter_exports_only_selected_columns(self):
        self.load_arange_dataset()

        callback = self.csv_logger(append=False, columns=['epoch', 'step'])
        trainer = TestTrainer(model=self.model,
                              callbacks=[callback],
                              logging_frecuency=5,
                              train_meters={'c': Averager()},
                              update_batch_fn=self.update_batch)

        trainer.train(self.train_dl, epochs=1)

        with open(self.stats_filename, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0], 'epoch,step\n')
            self.assertEqual(lines[1], '0,5\n')
            self.assertEqual(lines[2], '0,10')

    def test_csv_exporter_exports_empty_value_in_column_cell_if_associated_metric_does_not_exist(self):
        self.load_arange_dataset()

        callback = self.csv_logger(append=False, columns=['epoch', 'v', 'c'])
        trainer = TestTrainer(model=self.model,
                              callbacks=[callback],
                              logging_frecuency=5,
                              train_meters={'c': Averager()},
                              update_batch_fn=self.update_batch)

        trainer.train(self.train_dl, epochs=1)

        with open(self.stats_filename, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0], 'epoch,v,c\n')
            self.assertEqual(lines[1], '0,,2.0\n')
            self.assertEqual(lines[2], '0,,4.5')

    def test_repr(self):
        self.assertEqual(repr(CSVLogger(output='stats.csv',
                                        append=False,
                                        level='epoch')),
                         "CSVLogger(output='stats.csv', append=False, level='epoch')")
        self.assertEqual(repr(CSVLogger(output='stats.csv',
                                        append=True,
                                        level='step',
                                        columns=['train_acc'],
                                        hparams_columns=['lr'])),
                         "CSVLogger(output='stats.csv', append=True, level='step', columns=['train_acc'], hparams_columns=['lr'])")

    def tearDown(self):
        shutil.rmtree(self.base_tree)

class ProgbarTests(unittest.TestCase):
    # TODO: Don't rely in internal implentation to test the progress bar
    def setUp(self):
        self.model = nn.Linear(1, 1)
        self.callback = ProgbarLogger()

        def update_batch(trainer, x):
            trainer.train_meters['t_c'].measure(x.data[0][0])

        def validate_batch(validator, x):
            validator.meters['v_c'].measure(x.data[0][0])

        self.trainer = TestTrainer(model=self.model,
                              logging_frecuency=5,
                              train_meters={'t_c' : Averager()},
                              val_meters={'v_c' : Averager()},
                              update_batch_fn=update_batch,
                              callbacks=[self.callback],
                              valid_batch_fn=validate_batch)

    def test_progbar(self):
        X = torch.Tensor([1, 0.5, -0.5, -1]).view(-1, 1)
        train_dl = DataLoader(X, shuffle=False, batch_size=1)
        self.trainer.train(train_dl, epochs=2)
        self.assertEqual(self.callback.epoch_tqdm.unit, 'epoch')
        self.assertEqual(self.callback.epoch_tqdm.total, 2)

    def test_repr(self):
        self.assertEqual(repr(ProgbarLogger(ascii=True, notebook=False)),
                         'ProgbarLogger(ascii=True, notebook=False)')
        self.assertEqual(repr(ProgbarLogger(ascii=False, notebook=True, monitors=['train_acc'])),
                         'ProgbarLogger(ascii=False, notebook=True, monitors=[\'train_acc\'])')


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.model = nn.Linear(1, 1, bias=False)
        self.train_ds = None
        self.train_dl = None

        self.base_tree = os.path.join('tests', 'output')
        os.makedirs(self.base_tree)

        self.path = os.path.join(self.base_tree, 'checkpoint')

    def initialize_model_with_0(self):
        self.model.weight.data = torch.zeros(1, 1)

    def initialize_model_with_1(self):
        self.model.weight.data = torch.ones(1, 1)

    def initialize_model_with_random(self):
        self.model.weight.data.random_()

    def load_empty_dataset(self):
        self.train_ds = torch.Tensor([])
        self.train_dl = DataLoader(self.train_ds, batch_size=1, shuffle=False)

    def load_ones_dataset(self, batchs):
        self.train_ds = torch.ones(batchs, 1)
        self.train_dl = DataLoader(self.train_ds, batch_size=1, shuffle=False)

    def model_checkpoint(self, monitor, mode='min'):
        return ModelCheckpoint(self.path,
                               monitor=monitor,
                               mode=mode)

    @staticmethod
    def measure_zero():
        def update_batch_fn(trainer, _):
            trainer.train_meters['c'].measure(0)
        return update_batch_fn

    @staticmethod
    def meter_from_list(measure_list, meter_name):
        def update_batch_fn(trainer, _):
            trainer.train_meters['t'].measure(measure_list[trainer.epochs_trained])
            trainer.model.weight.data.add_(torch.ones(1, 1))
        return update_batch_fn

    def test_checkpoint_callback_doesnt_create_file_if_no_training(self):
        self.load_empty_dataset()

        checkpoint = self.model_checkpoint(monitor='c')
        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              train_meters={'c': Averager()},
                              logging_frecuency=1)

        trainer.train(self.train_dl, epochs=0)

        self.assertFalse(os.path.exists(self.path + '.zip'))
        self.assertEqual(os.listdir(self.path), [])

    def test_checkpoint_callback_doesnt_create_files_if_load_raises(self):
        checkpoint = self.model_checkpoint(monitor='c')
        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              train_meters={'c': Averager()},
                              logging_frecuency=1)

        try:
            checkpoint.load()
        except Exception:
            self.assertFalse(os.path.exists(self.path))

    def test_checkpoint_callback_raises_if_meter_not_found_in_meters_names(self):
        self.load_ones_dataset(2)
        checkpoint = self.model_checkpoint(monitor='xyz')
        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              logging_frecuency=1)

        try:
            trainer.train(self.train_dl, epochs=1)
            self.fail()
        except MeterNotFound as e:
            self.assertEqual(trainer.epochs_trained, 0)
        self.assertEqual(os.listdir(self.path), [])

    def test_checkpoint_callback_raises_if_meter_not_found(self):
        self.load_ones_dataset(2)
        checkpoint = self.model_checkpoint(monitor='t')
        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              train_meters={'t': Averager()},
                              logging_frecuency=1)

        try:
            trainer.train(self.train_dl, epochs=1)
            self.fail()
        except MeterNotFound as e:
            self.assertEqual(trainer.epochs_trained, 1)
        self.assertEqual(os.listdir(self.path), [])

    def test_checkpoint_callback_persist_model_on_first_trained_epoch(self):
        self.load_ones_dataset(2)
        checkpoint = self.model_checkpoint(monitor='c', mode='min')
        w = self.model.weight.data[0][0]

        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              train_meters={'c': Averager()},
                              update_batch_fn=self.measure_zero())

        trainer.train(self.train_dl, epochs=1)

        checkpoint = ModelCheckpoint(path=self.path, monitor='c', mode='min')

        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              update_batch_fn=self.measure_zero())

        self.initialize_model_with_random()
        data_best = checkpoint.load()

        self.assertEqual(data_best, {'epoch': 1, 'c': 0})
        self.assertEqual(self.model.weight.data[0][0], w)

    def test_checkpoint_callback_load_raises_if_metric_not_found(self):
        self.initialize_model_with_1()
        self.load_ones_dataset(2)
        checkpoint = self.model_checkpoint(monitor='c')

        w = self.model.weight.data[0][0]

        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              train_meters={'c': Averager()},
                              update_batch_fn=self.measure_zero())

        trainer.train(self.train_dl, epochs=1)

        checkpoint = self.model_checkpoint(monitor='xyz')
        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              update_batch_fn=self.measure_zero())

        self.initialize_model_with_0()

        try:
            checkpoint.load()
            self.fail()
        except MeterNotFound:
            self.assertEqual(self.model.weight.data[0][0], 0)

    def test_checkpoint_callback_not_persist_model_if_model_not_gets_better(self):
        self.initialize_model_with_0()
        self.load_ones_dataset(1)
        checkpoint = checkpoint = self.model_checkpoint(monitor='t', mode='min')

        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              train_meters={'t': Averager()},
                              update_batch_fn=self.meter_from_list([1, 2], 't'))

        trainer.train(self.train_dl, epochs=2)
        data_best = checkpoint.load()

        self.assertEqual(data_best, {'epoch':1, 't':1})
        self.assertEqual(self.model.weight.data[0][0], 1)

    def test_checkpoint_callback_repersist_model_if_model_gets_better(self):
        self.initialize_model_with_0()
        self.load_ones_dataset(1)
        checkpoint = self.model_checkpoint(monitor='t', mode='min')

        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              train_meters={'t': Averager()},
                              update_batch_fn=self.meter_from_list([2, 1], 't'))

        trainer.train(self.train_dl, epochs=2)
        data_best = checkpoint.load()

        self.assertEqual(data_best, {'epoch': 2, 't': 1})
        self.assertEqual(self.model.weight.data[0][0], 2)

    def test_checkpoint_callback_best_epoch_is_on_total_trained_epochs(self):
        self.initialize_model_with_0()
        self.load_ones_dataset(1)
        checkpoint = self.model_checkpoint(monitor='t', mode='min')

        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              train_meters={'t': Averager()},
                              update_batch_fn=self.meter_from_list([2, 3, 1], 't'))

        trainer.train(self.train_dl, epochs=2)
        trainer.train(self.train_dl, epochs=1)
        data_best = checkpoint.load()

        self.assertEqual(data_best, {'epoch': 3, 't': 1})
        self.assertEqual(self.model.weight.data[0][0], 3)

    def test_checkpoint_callback_load_epoch_reload_training_accuracy(self):
        self.load_ones_dataset(1)
        self.initialize_model_with_0()
        checkpoint = self.model_checkpoint(monitor='t', mode='min')

        measures = [2, 1, 3, 4]

        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              train_meters={'t': Averager()},
                              update_batch_fn=self.meter_from_list(measures, 't'))

        trainer.train(self.train_dl, epochs=3)

        checkpoint = self.model_checkpoint(monitor='t', mode='min')
        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              train_meters={'t': Averager()},
                              update_batch_fn=self.meter_from_list(measures, 't'))
        best = checkpoint.load()
        trainer.train(self.train_dl, epochs=1)
        data_best = checkpoint.load()

        self.assertEqual(data_best, {'epoch': 2, 't': 1})
        self.assertEqual(self.model.weight.data[0][0], 2)

    def test_checkpoint_callback_unrecognized_mode_raise_exception(self):
        self.assertRaises(Exception, lambda: ModelCheckpoint(path=self.path, monitor='c', mode='xyz'))

    def test_checkpoint_callback_with_max_mode_saves_model_on_maximum_monitor(self):
        self.initialize_model_with_0()
        self.load_ones_dataset(1)
        checkpoint = self.model_checkpoint(monitor='t', mode='max')

        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              train_meters={'t': Averager()},
                              update_batch_fn=self.meter_from_list([2, 3, 1], 't'))

        trainer.train(self.train_dl, epochs=3)
        data_best = checkpoint.load()

        self.assertEqual(data_best, {'epoch': 2, 't': 3})
        self.assertEqual(self.model.weight.data[0][0], 2)

    def tearDown(self):
        shutil.rmtree(self.base_tree)

    def test_mode_auto_infer_mode(self):
        self.load_ones_dataset(1)
        checkpoint = self.model_checkpoint(monitor='t', mode='auto')
        measures = []
        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              train_meters={'t': MSE()},
                              update_batch_fn=self.meter_from_list(measures, 't'))
        trainer.train(self.train_dl, epochs=0)
        self.assertEqual(checkpoint.mode, 'min')

        checkpoint = self.model_checkpoint(monitor='t', mode='auto')
        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              train_meters={'t': CategoricalAccuracy()},
                              update_batch_fn=self.meter_from_list(measures, 't'))
        trainer.train(self.train_dl, epochs=0)
        self.assertEqual(checkpoint.mode, 'max')

        checkpoint = self.model_checkpoint(monitor='t', mode='auto')
        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              train_meters={'t': Averager()},
                              update_batch_fn=self.meter_from_list(measures, 't'))
        try:
            trainer.train(self.train_dl, epochs=0)
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), torchero.utils.defaults.INVALID_MODE_INFERENCE_MESSAGE.format(meter='Averager'))

class EarlyStoppingTests(unittest.TestCase):
    def setUp(self):
        self.load_model()

    def load_model(self):
        self.model = DummyModel()

    def load_ones_dataset(self, batchs):
        self.train_ds = torch.ones(batchs, 1)
        self.train_dl = DataLoader(self.train_ds, batch_size=1, shuffle=False)

    def early_callback(self, monitor, mode='min', patience=0, min_delta=0):
        return EarlyStopping(monitor=monitor,
                               mode=mode,
                               patience=patience,
                               min_delta=min_delta)

    def test_repr(self):
        self.assertEqual(repr(EarlyStopping(monitor='train_acc', min_delta=0.1, patience=1, mode='min')),
                         "EarlyStopping(monitor='train_acc', min_delta=0.1, patience=1, mode='min')")

    @staticmethod
    def meter_from_list(measure_list, meter_name):
        def update_batch_fn(trainer, _):
            trainer.train_meters['t'].measure(measure_list[trainer.epochs_trained])
        return update_batch_fn

    def make_test_trainer(self, callbacks, measures):
        return TestTrainer(model=self.model,
                           callbacks=callbacks,
                           logging_frecuency=1,
                           train_meters={'t': Averager()},
                           update_batch_fn=self.meter_from_list(measures, 't'))

    def test_never_stop_if_measure_is_always_decreasing(self):
        self.load_ones_dataset(1)
        callback_min = self.early_callback(monitor='t', mode='min')
        callback_max = self.early_callback(monitor='t', mode='max')

        measures_min = [3, 2, 1, 0]
        measures_max = [0, 1, 2, 3]

        trainer_min = self.make_test_trainer(callbacks=[callback_min], measures=measures_min)
        trainer_max = self.make_test_trainer(callbacks=[callback_max], measures=measures_max)

        trainer_min.train(self.train_dl, epochs=4)
        trainer_max.train(self.train_dl, epochs=4)
        self.assertEqual(trainer_min.epochs_trained, 4)
        self.assertEqual(trainer_max.epochs_trained, 4)

    def test_stops_when_measure_not_strictly_improve_with_0_patience(self):
        self.load_ones_dataset(1)
        callback_min = self.early_callback(monitor='t', mode='min')
        callback_max = self.early_callback(monitor='t', mode='max')

        measures_min = [3, 3, 2]
        measures_max = [3, 3, 4]

        trainer_min = self.make_test_trainer(callbacks=[callback_min], measures=measures_min)
        trainer_max = self.make_test_trainer(callbacks=[callback_max], measures=measures_max)

        trainer_min.train(self.train_dl, epochs=3)
        trainer_max.train(self.train_dl, epochs=3)
        self.assertEqual(trainer_min.epochs_trained, 2)
        self.assertEqual(trainer_max.epochs_trained, 2)

    def test_stops_when_measure_not_improve_with_0_patience(self):
        self.load_ones_dataset(1)
        callback_min = self.early_callback(monitor='t', mode='min')
        callback_max = self.early_callback(monitor='t', mode='max')

        measures_min = [3, 4, 1]
        measures_max = [2, 1, 4]

        trainer_min = self.make_test_trainer(callbacks=[callback_min], measures=measures_min)
        trainer_max = self.make_test_trainer(callbacks=[callback_max], measures=measures_max)

        trainer_min.train(self.train_dl, epochs=3)
        trainer_max.train(self.train_dl, epochs=3)
        self.assertEqual(trainer_min.epochs_trained, 2)
        self.assertEqual(trainer_max.epochs_trained, 2)

    def test_stops_when_measure_not_improve_more_than_1_epoch_of_margin_with_1_patience(self):
        self.load_ones_dataset(1)
        callback_min = self.early_callback(monitor='t', mode='min', patience=1)
        callback_max = self.early_callback(monitor='t', mode='max', patience=1)

        measures_min = [3, 4, 2, 0]
        measures_max = [3, 2, 4, 5]

        trainer_min = self.make_test_trainer(callbacks=[callback_min], measures=measures_min)
        trainer_max = self.make_test_trainer(callbacks=[callback_max], measures=measures_max)

        trainer_min.train(self.train_dl, epochs=4)
        trainer_max.train(self.train_dl, epochs=4)
        self.assertEqual(trainer_min.epochs_trained, 4)
        self.assertEqual(trainer_max.epochs_trained, 4)

    def test_should_reset_the_patience_counter_after_improvement(self):
        self.load_ones_dataset(1)
        callback_min = self.early_callback(monitor='t', mode='min', patience=1)
        callback_max = self.early_callback(monitor='t', mode='max', patience=1)

        measures_min = [3, 4, 2, 3, 1]
        measures_max = [3, 2, 4, 3, 5]

        trainer_min = self.make_test_trainer(callbacks=[callback_min], measures=measures_min)
        trainer_max = self.make_test_trainer(callbacks=[callback_max], measures=measures_max)

        trainer_min.train(self.train_dl, epochs=5)
        trainer_max.train(self.train_dl, epochs=5)
        self.assertEqual(trainer_min.epochs_trained, 5)
        self.assertEqual(trainer_max.epochs_trained, 5)

    def test_min_delta_param_should_set_different_margin_for_error_improvement_with_max(self):
        self.load_ones_dataset(1)
        callback_min = self.early_callback(monitor='t', mode='min', patience=0, min_delta=1)
        callback_max = self.early_callback(monitor='t', mode='max', patience=0, min_delta=1)

        measures_min = [3, 3.99, 2, 4, 1]
        measures_max = [3, 2.01, 4, 2, 5]

        trainer_min = self.make_test_trainer(callbacks=[callback_min], measures=measures_min)
        trainer_max = self.make_test_trainer(callbacks=[callback_max], measures=measures_max)

        trainer_min.train(self.train_dl, epochs=5)
        trainer_max.train(self.train_dl, epochs=5)
        self.assertEqual(trainer_min.epochs_trained, 2)
        self.assertEqual(trainer_max.epochs_trained, 2)

    def test_max_min_mode_are_the_only_modes_available(self):
        mode = 'maximum'
        try:
            checkpoint = self.early_callback(monitor='t', mode=mode)
            self.fail()
        except ValueError as e:
            self.assertEqual(str(e), EarlyStopping.UNRECOGNIZED_MODE_MESSAGE.format(mode=mode))

    def test_mode_auto_infer_mode(self):
        self.load_ones_dataset(1)
        callback = self.early_callback(monitor='v', mode='auto', patience=0, min_delta=1)
        self.assertEqual(callback.monitor, 'v')
        self.assertEqual(callback.mode, 'auto')
        self.assertEqual(callback.patience, 0)
        self.assertEqual(callback.min_delta, 1)
        measures = []
        trainer = TestTrainer(model=self.model,
                              callbacks=[callback],
                              logging_frecuency=1,
                              train_meters={'v': MSE()},
                              update_batch_fn=self.meter_from_list(measures, 'v'))
        trainer.train(self.train_dl, epochs=0)
        self.assertEqual(callback.mode, 'min')

        callback = self.early_callback(monitor='v', mode='auto', patience=0, min_delta=1)
        trainer = TestTrainer(model=self.model,
                              callbacks=[callback],
                              logging_frecuency=1,
                              train_meters={'v': CategoricalAccuracy()},
                              update_batch_fn=self.meter_from_list(measures, 'v'))
        trainer.train(self.train_dl, epochs=0)
        self.assertEqual(callback.mode, 'max')

        callback = self.early_callback(monitor='v', mode='auto', patience=0, min_delta=1)
        trainer = TestTrainer(model=self.model,
                              callbacks=[callback],
                              logging_frecuency=1,
                              train_meters={'v': Averager()},
                              update_batch_fn=self.meter_from_list(measures, 'v'))
        try:
            trainer.train(self.train_dl, epochs=0)
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), torchero.utils.defaults.INVALID_MODE_INFERENCE_MESSAGE.format(meter='Averager'))
