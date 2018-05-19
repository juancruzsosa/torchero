import shutil
from .common import *


class HistoryCallbackTests(unittest.TestCase):
    def test_history_callback_register_every_training_stat(self):
        self.model = DummyModel()
        train_dataset = torch.arange(10).view(-1, 1)
        valid_dataset = torch.arange(10).view(-1, 1)

        train_dl = DataLoader(train_dataset, shuffle=False, batch_size=1)
        valid_dl = DataLoader(valid_dataset, shuffle=False, batch_size=1)

        def update_batch(trainer, x):
            trainer.train_meters['t_c'].measure(x.data[0][0])

        def validate_batch(validator, x):
            validator.meters['v_c'].measure(x.data[0][0])

        trainer = TestTrainer(model=self.model,
                              logging_frecuency=5,
                              train_meters={'t_c' : Averager()},
                              val_meters={'v_c' : Averager()},
                              update_batch_fn=update_batch,
                              valid_batch_fn=validate_batch)
        self.assertEqual(trainer.metrics, {})

        self.assertEqual(set(trainer.meters_names()), set(['t_c', 'v_c']))

        trainer.train(train_dl, valid_dataloader=valid_dl, epochs=1)

        expected_registry = [{'epoch': 0, 'step': 4, 't_c': 2.0, 'v_c': 4.5},
                             {'epoch': 0, 'step': 9, 't_c': 7.0, 'v_c': 4.5}]

        self.assertEqual(list(trainer.history), expected_registry)
        self.assertEqual(trainer.metrics, {'t_c': 7.0, 'v_c': 4.5})


class CSVExporterTests(unittest.TestCase):
    def setUp(self):
        self.base_tree = os.path.join('tests', 'output')
        self.load_model()
        os.makedirs(self.base_tree)

        self.stats_filename = os.path.join(self.base_tree, 'stats.csv')

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
        return CSVLogger(output=self.stats_filename, append=append, columns=columns)

    def test_csv_exporter_print_header_at_begining_of_training(self):
        self.load_empty_dataset()

        callback = self.csv_logger(append=False)
        self.assertFalse(os.path.exists(self.stats_filename))

        trainer = TestTrainer(model=self.model,
                              callbacks=[callback],
                              train_meters={'c' : Averager()},
                              logging_frecuency=5,
                              update_batch_fn=self.update_batch)

        trainer.train(self.train_dl, epochs=1)

        with open(self.stats_filename, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0], 'epoch,step,c')

    def test_csv_exporter_stats_write_stats_to_csv_after_every_log(self):
        self.load_arange_dataset()

        callback = self.csv_logger(append=False)
        self.assertFalse(os.path.exists(self.stats_filename))

        trainer = TestTrainer(model=self.model,
                              callbacks=[callback],
                              logging_frecuency=5,
                              train_meters={'c': Averager()},
                              update_batch_fn=self.update_batch)
        trainer.train(self.train_dl, epochs=1)

        with open(self.stats_filename, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0], 'epoch,step,c\n')
            self.assertEqual(lines[1], '0,4,2.0\n')
            self.assertEqual(lines[2], '0,9,7.0')



    def test_export_stats_can_append_stats_with_matching_cols_to_previous_training(self):
        self.load_arange_dataset()

        callback = self.csv_logger(append=True)
        self.assertFalse(os.path.exists(self.stats_filename))

        trainer = TestTrainer(model=self.model,
                              callbacks=[callback],
                              logging_frecuency=5,
                              train_meters={'c': Averager()},
                              update_batch_fn=self.update_batch)

        trainer.train(self.train_dl, epochs=1)
        trainer.train(self.train_dl, epochs=1)

        with open(self.stats_filename, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 5)
            self.assertEqual(lines[0], 'epoch,step,c\n')
            self.assertEqual(lines[1], '0,4,2.0\n')
            self.assertEqual(lines[2], '0,9,7.0\n')
            self.assertEqual(lines[3], '1,4,2.0\n')
            self.assertEqual(lines[4], '1,9,7.0')

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
            self.assertEqual(lines[1], '0,4\n')
            self.assertEqual(lines[2], '0,9')

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
            self.assertEqual(lines[2], '0,,7.0')

    def test_csv_exporter_overwrite_entire_file_if_append_is_false(self):
        self.load_arange_dataset()

        callback = self.csv_logger(append=False)

        trainer = TestTrainer(model=self.model,
                              callbacks=[callback],
                              logging_frecuency=5)

        trainer.train(self.train_dl, epochs=1)
        trainer.train(self.train_dl, epochs=2)

        with open(self.stats_filename, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 5)
            self.assertEqual(lines[0], 'epoch,step\n')
            self.assertEqual(lines[1], '1,4\n')
            self.assertEqual(lines[2], '1,9\n')
            self.assertEqual(lines[3], '2,4\n')
            self.assertEqual(lines[4], '2,9')

    def tearDown(self):
        shutil.rmtree(self.base_tree)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.model = nn.Linear(1, 1, bias=False)
        self.train_ds = None
        self.train_dl = None

        self.base_tree = os.path.join('tests', 'output')
        os.makedirs(self.base_tree)

        self.temp_dir = os.path.join(self.base_tree, 'tmp')
        os.makedirs(self.temp_dir)

        self.checkpoint_file = os.path.join(self.base_tree, 'checkpoint')

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
        return ModelCheckpoint(self.checkpoint_file,
                               monitor=monitor,
                               mode=mode,
                               temp_dir=self.temp_dir)

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

        self.assertFalse(os.path.exists(self.checkpoint_file + '.zip'))
        self.assertEqual(os.listdir(self.temp_dir), [])

    def test_checkpoint_callback_doesnt_create_files_if_load_raises(self):
        checkpoint = self.model_checkpoint(monitor='c')
        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              train_meters={'c': Averager()},
                              logging_frecuency=1)

        try:
            checkpoint.load()
        except Exception:
            self.assertFalse(os.path.exists(self.checkpoint_file + '.zip'))
            self.assertEqual(os.listdir(self.temp_dir), [])

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
        self.assertEqual(os.listdir(self.temp_dir), [])

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
        self.assertEqual(os.listdir(self.temp_dir), [])

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

        checkpoint = ModelCheckpoint(path=self.checkpoint_file, monitor='c', mode='min', temp_dir=self.temp_dir)

        trainer = TestTrainer(model=self.model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              update_batch_fn=self.measure_zero())

        self.initialize_model_with_random()
        data_best = checkpoint.load()

        self.assertEqual(data_best, {'epoch': 1, 'c': 0})
        self.assertEqual(self.model.weight.data[0][0], w)
        self.assertEqual(os.listdir(self.temp_dir), [])

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

        self.assertEqual(os.listdir(self.temp_dir), [])

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
        self.assertRaises(Exception, lambda: ModelCheckpoint(path=self.checkpoint_file, monitor='c', mode='xyz'))

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
