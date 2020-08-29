from torch.optim import Adam
from torch import nn
from .common import *
from torchero import SupervisedTrainer
from torchero.hparams import OptimP

class HparamsTest(unittest.TestCase):
    def setUp(self):
        self.model = nn.Linear(2, 1)

    def test_str_hparams(self):
        trainer = SupervisedTrainer(self.model, 'mse', 'adam', hparams={'model_name': 'linear_model'})
        self.assertEqual(trainer.hparams['model_name'], 'linear_model')

    def test_lambda_hparam_should_evaluate_function(self):
        trainer = SupervisedTrainer(self.model, 'mse', 'adam', hparams={'model_name': 'linear_model',
                                                                        'optimizer_name': lambda trainer: trainer.optimizer.__class__.__name__})
        self.assertEqual(trainer.hparams['model_name'], 'linear_model')
        self.assertEqual(trainer.hparams['optimizer_name'], 'Adam')
        self.assertEqual(trainer.hparams.get('optimizer_name'), 'Adam')
        self.assertEqual(trainer.hparams.get('unexistent', None), None)

class OptimParamsTests(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))

    def test_lr_hparam_should_return_lr(self):
        trainer = SupervisedTrainer(self.model,
                                    'mse',
                                    Adam(self.model.parameters(), lr=2, eps=0.1),
                                    hparams={'model_name': 'linear_model',
                                             'optimizer_name': OptimP('name'),
                                             'lr': OptimP('lr'),
                                             'eps': OptimP('eps')})
        self.assertEqual(trainer.hparams['model_name'], 'linear_model')
        self.assertEqual(trainer.hparams['optimizer_name'], 'adam')
        self.assertEqual(trainer.hparams['lr'], 2)
        self.assertEqual(trainer.hparams['eps'], 0.1)
        self.assertEqual(trainer.hparams.get('optimizer_name'), 'adam')
        self.assertEqual(trainer.hparams.get('unexistent', None), None)

    def test_lr_hparam_can_be_setted(self):
        trainer = SupervisedTrainer(self.model,
                                    'mse',
                                    Adam(self.model.parameters(), lr=2, eps=0.1),
                                    hparams={'lr': OptimP('lr'),
                                             'optimizer_name': OptimP('name')})
        param_group = next(iter(trainer.optimizer.param_groups))
        param_group['lr'] = 1
        self.assertEqual(trainer.hparams['lr'], 1)
        trainer.hparams['lr'] = 0.5
        trainer.hparams['optimizer_name'] = 'Adam'
        self.assertEqual(trainer.hparams['lr'], 0.5)
        self.assertEqual(trainer.hparams['optimizer_name'], 'Adam')
        self.assertEqual(param_group['lr'], 0.5)

    def test_params_can_be_indexed(self):
        trainer = SupervisedTrainer(self.model,
                                    'mse',
                                    Adam([{'params': self.model[0].parameters(), 'lr':2, 'eps': 0.1},
                                          {'params': self.model[1].parameters(), 'lr':1, 'eps': 0.2}]),
                                    hparams={'lr_1': OptimP('lr.0'),
                                             'lr_2': OptimP('lr.1'),
                                             'eps_1': OptimP('eps.0'),
                                             'eps_2': OptimP('eps.1')})
        s = list(trainer.optimizer.param_groups)
        self.assertEqual(trainer.hparams['lr_1'], 2)
        self.assertEqual(trainer.hparams['lr_2'], 1)
        self.assertEqual(trainer.hparams['eps_1'], 0.1)
        self.assertEqual(trainer.hparams['eps_2'], 0.2)

    def test_params_can_not_be_created_from_unexistent_properties(self):
        try:
            trainer = SupervisedTrainer(self.model,
                                        'mse',
                                        'adam',
                                        hparams={'lr': OptimP('momentum')})
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), "Adam optimizer has not property 'momentum'!")

        try:
            trainer = SupervisedTrainer(self.model,
                                        'mse',
                                        Adam([{'params': self.model[0].parameters(), 'lr':2, 'eps': 0.1},
                                              {'params': self.model[1].parameters(), 'lr':1, 'eps': 0.2}]),
                                        hparams={'lr': OptimP('lr.5')})
        except IndexError as e:
            self.assertEqual(str(e), "Adam optimizer has not #5 param group")


if __name__ == '__main__':
    unittest.main()
