import sys
import shutil
from .common import *

class CallbacksTests(unittest.TestCase):
    def setUp(self):
        self.base_tree = os.path.join('tests', 'output')
        os.makedirs(self.base_tree)

        self.temp_dir = os.path.join(self.base_tree, 'tmp')
        os.makedirs(self.temp_dir)

        self.stats_filename = os.path.join(self.base_tree, 'stats.csv')
        self.checkpoint_file = os.path.join(self.base_tree, 'checkpoint')

    def test_history_callback_register_every_training_stat(self):
        self.model = DummyModel()
        train_dataset = torch.arange(10).view(-1, 1)
        valid_dataset = torch.arange(10).view(-1, 1)

        train_dl = DataLoader(train_dataset, shuffle=False, batch_size=1)
        valid_dl = DataLoader(valid_dataset, shuffle=False, batch_size=1)
        callback = History()

        def update_batch(trainer, x):
            trainer.meters['t_c'].measure(x.data[0][0])

        def validate_batch(trainer, x):
            trainer.meters['v_c'].measure(x.data[0][0])

        trainer = TestTrainer(model=self.model,
                              callbacks=[callback],
                              logging_frecuency=5,
                              meters={'t_c' : Averager(),
                                      'v_c' : Averager()},
                              update_batch_fn=update_batch,
                              valid_batch_fn=validate_batch)
        self.assertEqual(trainer.last_stats, {})

        self.assertEqual(set(trainer.meters_names()), set(['t_c', 'v_c']))

        trainer.train(train_dl, valid_dataloader=valid_dl, epochs=1)

        expected_registry = [{'epoch': 0, 'step': 4, 't_c': 2.0, 'v_c': 4.5},
                             {'epoch': 0, 'step': 9, 't_c': 7.0, 'v_c': 4.5}]

        self.assertEqual(callback.registry, expected_registry)
        self.assertEqual(trainer.last_stats, {'t_c': 7.0, 'v_c': 4.5})

    def test_csv_exporter_print_header_at_begining_of_training(self):
        self.model = DummyModel()
        dataset = torch.Tensor([])

        dl = DataLoader(dataset, shuffle=False, batch_size=1)
        callback = CSVLogger(output=self.stats_filename, append=False)
        self.assertFalse(os.path.exists(self.stats_filename))

        def update_batch(trainer, x):
            trainer.meters['c'].measure(x.data[0][0])

        trainer = TestTrainer(model=self.model,
                              callbacks=[callback],
                              meters={'c' : Averager()},
                              logging_frecuency=5,
                              update_batch_fn=update_batch)

        trainer.train(dl, epochs=1)

        with open(self.stats_filename, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0], 'epoch,step,c')

    def test_csv_exporter_stats_write_stats_to_csv_after_every_log(self):
        self.model = DummyModel()
        dataset = torch.arange(10).view(-1, 1)

        dl = DataLoader(dataset, shuffle=False, batch_size=1)
        callback = CSVLogger(output=self.stats_filename, append=False)
        self.assertFalse(os.path.exists(self.stats_filename))

        def update_batch(trainer, x):
            trainer.meters['c'].measure(x.data[0][0])

        trainer = TestTrainer(model=self.model,
                              callbacks=[callback],
                              logging_frecuency=5,
                              meters={'c' : Averager()},
                              update_batch_fn=update_batch)

        trainer.train(dl, epochs=1)

        with open(self.stats_filename, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0], 'epoch,step,c\n')
            self.assertEqual(lines[1], '0,4,2.0\n')
            self.assertEqual(lines[2], '0,9,7.0')

    def test_export_stats_can_append_stats_with_matching_cols_to_previous_training(self):
        self.model = DummyModel()
        dataset = torch.arange(10).view(-1, 1)

        dl = DataLoader(dataset, shuffle=False, batch_size=1)
        callback = CSVLogger(output=self.stats_filename, append=True)
        self.assertFalse(os.path.exists(self.stats_filename))

        def update_batch(trainer, x):
            trainer.meters['c'].measure(x.data[0][0])

        trainer = TestTrainer(model=self.model,
                              callbacks=[callback],
                              logging_frecuency=5,
                              meters={'c': Averager()},
                              update_batch_fn=update_batch)

        trainer.train(dl, epochs=1)
        trainer.train(dl, epochs=1)

        with open(self.stats_filename, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 5)
            self.assertEqual(lines[0], 'epoch,step,c\n')
            self.assertEqual(lines[1], '0,4,2.0\n')
            self.assertEqual(lines[2], '0,9,7.0\n')
            self.assertEqual(lines[3], '1,4,2.0\n')
            self.assertEqual(lines[4], '1,9,7.0')

    def test_csv_exporter_exports_only_selected_columns(self):
        self.model = DummyModel()
        dataset = torch.arange(10).view(-1, 1)

        dl = DataLoader(dataset, shuffle=False, batch_size=1)
        callback = CSVLogger(output=self.stats_filename, append=False, columns=['epoch', 'step'])

        def update_batch(trainer, x):
            trainer.meters['c'].measure(x.data[0][0])

        trainer = TestTrainer(model=self.model,
                              callbacks=[callback],
                              logging_frecuency=5,
                              meters={'c': Averager()},
                              update_batch_fn=update_batch)

        trainer.train(dl, epochs=1)

        with open(self.stats_filename, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0], 'epoch,step\n')
            self.assertEqual(lines[1], '0,4\n')
            self.assertEqual(lines[2], '0,9')

    def test_csv_exporter_exports_empty_value_in_column_cell_if_associated_metric_does_not_exist(self):
        self.model = DummyModel()
        dataset = torch.arange(10).view(-1, 1)

        dl = DataLoader(dataset, shuffle=False, batch_size=1)
        callback = CSVLogger(output=self.stats_filename, append=False, columns=['epoch', 'v', 't'])

        def update_batch(trainer, x):
            trainer.meters['t'].measure(x.data[0][0])

        trainer = TestTrainer(model=self.model,
                              callbacks=[callback],
                              logging_frecuency=5,
                              meters={'t': Averager()},
                              update_batch_fn=update_batch)

        trainer.train(dl, epochs=1)

        with open(self.stats_filename, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0], 'epoch,v,t\n')
            self.assertEqual(lines[1], '0,,2.0\n')
            self.assertEqual(lines[2], '0,,7.0')

    def test_csv_exporter_overwrite_entire_file_if_append_is_false(self):
        self.model = DummyModel()
        dataset = torch.arange(10).view(-1, 1)

        dl = DataLoader(dataset, shuffle=False, batch_size=1)
        callback = CSVLogger(output=self.stats_filename, append=False)

        trainer = TestTrainer(model=self.model,
                              callbacks=[callback],
                              logging_frecuency=5)

        trainer.train(dl, epochs=1)
        trainer.train(dl, epochs=2)

        with open(self.stats_filename, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 5)
            self.assertEqual(lines[0], 'epoch,step\n')
            self.assertEqual(lines[1], '1,4\n')
            self.assertEqual(lines[2], '1,9\n')
            self.assertEqual(lines[3], '2,4\n')
            self.assertEqual(lines[4], '2,9')

    def test_checkpoint_callback_doesnt_create_file_if_no_training(self):
        model = DummyModel()
        checkpoint = ModelCheckpoint(path=self.checkpoint_file, monitor='c')
        dataset = torch.Tensor([])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        trainer = TestTrainer(model=model,
                              callbacks=[checkpoint],
                              meters = {'c' : Averager()},
                              logging_frecuency=1)

        trainer.train(dataloader, epochs=0)
        self.assertFalse(os.path.exists(self.checkpoint_file + '.zip'))
        self.assertEqual(os.listdir(self.temp_dir), [])

    def test_checkpoint_callback_doesnt_create_files_if_load_raises(self):
        model = DummyModel()
        checkpoint = ModelCheckpoint(path=self.checkpoint_file, monitor='c')
        dataset = torch.Tensor([])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        trainer = TestTrainer(model=model,
                              callbacks=[checkpoint],
                              meters={'c' : Averager()},
                              logging_frecuency=1)

        try:
            checkpoint.load()
        except Exception:
            self.assertFalse(os.path.exists(self.checkpoint_file + '.zip'))
            self.assertEqual(os.listdir(self.temp_dir), [])

    def test_checkpoint_callback_raises_if_meter_not_found_in_meters_names(self):
        model = DummyModel()
        checkpoint = ModelCheckpoint(path=self.checkpoint_file, monitor='xyz', temp_dir=self.temp_dir)
        dataset = torch.ones(1, 2)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        trainer = TestTrainer(model=model,
                              callbacks=[checkpoint],
                              logging_frecuency=1)
        try:
            trainer.train(dataloader, epochs=1)
            self.fail()
        except MeterNotFound as e:
            self.assertEqual(trainer.epochs_trained, 0)
        self.assertEqual(os.listdir(self.temp_dir), [])

    def test_checkpoint_callback_raises_if_meter_not_found(self):
        model = DummyModel()
        checkpoint = ModelCheckpoint(path=self.checkpoint_file, monitor='t', temp_dir=self.temp_dir)
        dataset = torch.ones(1, 2)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        trainer = TestTrainer(model=model,
                              callbacks=[checkpoint],
                              meters={'t': Averager()},
                              logging_frecuency=1)

        try:
            trainer.train(dataloader, epochs=1)
            self.fail()
        except MeterNotFound as e:
            self.assertEqual(trainer.epochs_trained, 1)
        self.assertEqual(os.listdir(self.temp_dir), [])

    def test_checkpoint_callback_persist_model_on_first_trained_epoch(self):
        model = nn.Linear(1, 1, bias=False)
        w = model.weight.data[0][0]
        checkpoint = ModelCheckpoint(path=self.checkpoint_file, monitor='c', mode='min', temp_dir=self.temp_dir)
        dataset = torch.ones(2, 1)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        def update_batch(trainer, x):
            trainer.meters['c'].measure(0)

        trainer = TestTrainer(model=model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              meters={'c' : Averager()},
                              update_batch_fn=update_batch)

        trainer.train(dataloader, epochs=1)

        checkpoint = ModelCheckpoint(path=self.checkpoint_file, monitor='c', mode='min', temp_dir=self.temp_dir)
        trainer = TestTrainer(model=model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              update_batch_fn=update_batch)
        model.weight.data.random_()
        data_best = checkpoint.load()
        self.assertEqual(data_best, {'epoch':1, 'c':0})
        self.assertEqual(model.weight.data[0][0], w)
        self.assertEqual(os.listdir(self.temp_dir), [])

    def test_checkpoint_callback_load_raises_if_metric_not_found(self):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data = torch.ones(1,1)
        w = model.weight.data[0][0]
        checkpoint = ModelCheckpoint(path=self.checkpoint_file, monitor='c', temp_dir=self.temp_dir)
        dataset = torch.ones(2, 1)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        def update_batch(trainer, x):
            trainer.meters['c'].measure(0)

        trainer = TestTrainer(model=model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              meters={'c' : Averager()},
                              update_batch_fn=update_batch)

        trainer.train(dataloader, epochs=1)

        checkpoint = ModelCheckpoint(path=self.checkpoint_file, monitor='xyz', temp_dir=self.temp_dir)
        trainer = TestTrainer(model=model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              update_batch_fn=update_batch)

        model.weight.data = torch.zeros(1,1)

        try:
            checkpoint.load()
            self.fail()
        except MeterNotFound:
            self.assertEqual(model.weight.data[0][0], 0)

        self.assertEqual(os.listdir(self.temp_dir), [])

    def test_checkpoint_callback_not_persist_model_if_model_not_gets_better(self):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data = torch.zeros(1,1)
        checkpoint = ModelCheckpoint(path=self.checkpoint_file, monitor='t', mode='min', temp_dir=self.temp_dir)
        dataset = torch.ones(1, 1)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        measures = [1, 2]

        def update_batch(trainer, x):
            trainer.meters['t'].measure(measures[trainer.epoch])
            trainer.model.weight.data.add_(torch.ones(1,1))

        trainer = TestTrainer(model=model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              meters={'t' : Averager()},
                              update_batch_fn=update_batch)
        trainer.train(dataloader, epochs=2)
        data_best = checkpoint.load()

        self.assertEqual(data_best, {'epoch':1, 't':1})
        self.assertEqual(model.weight.data[0][0], 1)

    def test_checkpoint_callback_repersist_model_if_model_gets_better(self):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data = torch.zeros(1,1)
        checkpoint = ModelCheckpoint(path=self.checkpoint_file, monitor='t', mode='min', temp_dir=self.temp_dir)
        dataset = torch.ones(1, 1)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        measures = [2, 1]

        def update_batch(trainer, x):
            trainer.meters['t'].measure(measures[trainer.epoch])
            trainer.model.weight.data.add_(torch.ones(1,1))

        trainer = TestTrainer(model=model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              meters={'t' : Averager()},
                              update_batch_fn=update_batch)
        trainer.train(dataloader, epochs=2)
        data_best = checkpoint.load()

        self.assertEqual(data_best, {'epoch':2, 't':1})
        self.assertEqual(model.weight.data[0][0], 2)

    def test_checkpoint_callback_best_epoch_is_on_total_trained_epochs(self):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data = torch.zeros(1,1)
        checkpoint = ModelCheckpoint(path=self.checkpoint_file, monitor='t', mode='min', temp_dir=self.temp_dir)
        dataset = torch.ones(1, 1)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        measures = [2, 3, 1]

        def update_batch(trainer, x):
            trainer.meters['t'].measure(measures[trainer.epochs_trained])
            trainer.model.weight.data.add_(torch.ones(1,1))

        trainer = TestTrainer(model=model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              meters={'t' : Averager()},
                              update_batch_fn=update_batch)
        trainer.train(dataloader, epochs=2)
        trainer.train(dataloader, epochs=1)
        data_best = checkpoint.load()

        self.assertEqual(data_best, {'epoch':3, 't':1})
        self.assertEqual(model.weight.data[0][0], 3)

    def test_checkpoint_callback_load_epoch_reload_training_accuracy(self):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data = torch.zeros(1,1)
        checkpoint = ModelCheckpoint(path=self.checkpoint_file, monitor='t', mode='min', temp_dir=self.temp_dir)
        dataset = torch.ones(1, 1)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        measures = [2, 1, 3, 4]

        def update_batch(trainer, x):
            trainer.meters['t'].measure(measures[trainer.epochs_trained])
            trainer.model.weight.data.add_(torch.ones(1,1))

        trainer = TestTrainer(model=model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              meters={'t' : Averager()},
                              update_batch_fn=update_batch)
        trainer.train(dataloader, epochs=3)
        checkpoint = ModelCheckpoint(path=self.checkpoint_file, monitor='t', mode='min', temp_dir=self.temp_dir)
        trainer = TestTrainer(model=model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              meters={'t' : Averager()},
                              update_batch_fn=update_batch)
        best = checkpoint.load()
        trainer.train(dataloader, epochs=1)
        data_best = checkpoint.load()

        self.assertEqual(data_best, {'epoch':2, 't':1})
        self.assertEqual(model.weight.data[0][0], 2)

    def test_checkpoint_callback_unrecognized_mode_raise_exception(self):
        self.assertRaises(Exception, lambda:ModelCheckpoint(path=self.checkpoint_file, monitor='c', mode='xyz'))

    def test_checkpoint_callback_with_max_mode_saves_model_on_maximum_monitor(self):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data = torch.zeros(1,1)
        checkpoint = ModelCheckpoint(path=self.checkpoint_file, monitor='t', mode='max', temp_dir=self.temp_dir)
        dataset = torch.ones(1, 1)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        measures = [2, 3, 1]

        def update_batch(trainer, x):
            trainer.meters['t'].measure(measures[trainer.epochs_trained])
            trainer.model.weight.data.add_(torch.ones(1,1))

        trainer = TestTrainer(model=model,
                              callbacks=[checkpoint],
                              logging_frecuency=1,
                              meters={'t' : Averager()},
                              update_batch_fn=update_batch)
        trainer.train(dataloader, epochs=3)
        data_best = checkpoint.load()

        self.assertEqual(data_best, {'epoch':2, 't':3})
        self.assertEqual(model.weight.data[0][0], 2)

    def tearDown(self):
        shutil.rmtree(self.base_tree)
