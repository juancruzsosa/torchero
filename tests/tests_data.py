import tempfile
import json

import pandas as pd

import torch
import torchero
from .common import *
from torchero.utils.data import CrossFoldValidation, train_test_split
from torchero.utils.data.datasets import UnsuperviseDataset, \
                                             ShrinkDataset
from torchero.utils.text.datasets import TextClassificationDataset

class TestsDataUtils(unittest.TestCase):
    def assertDatasetEquals(self, a, b):
        self.assertEqual(list(a), list(b))

    def assertStopIteration(self, it):
        try:
            next(it)
            self.fail()
        except Exception as e:
            self.assertIsInstance(e, StopIteration)

    def setUp(self):
        torch.manual_seed(1)

    def test_can_not_split_when_dataset_is_empty(self):
        train_dataset = []

        try:
            CrossFoldValidation(dataset=train_dataset, valid_size=-0.01)
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), CrossFoldValidation.INVALID_VALID_SIZE_MESSAGE)

    def test_can_not_split_when_valid_size_is_less_than_zero(self):
        train_dataset = [1, 2, 3]

        try:
            CrossFoldValidation(dataset=train_dataset, valid_size=-0.01)
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), CrossFoldValidation.INVALID_VALID_SIZE_MESSAGE)

    def test_can_not_split_when_valid_size_greater_than_one(self):
        train_dataset = [1, 2, 3]

        try:
            CrossFoldValidation(dataset=train_dataset, valid_size=1.01)
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), CrossFoldValidation.INVALID_VALID_SIZE_MESSAGE)

    def test_can_not_split_when_valid_size_is_zero(self):
        train_dataset = [1, 2, 3, 4]

        try:
            CrossFoldValidation(dataset=train_dataset, valid_size=0)
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), CrossFoldValidation.INVALID_VALID_SIZE_MESSAGE)

    def test_can_not_split_when_valid_size_is_less_than_one_sample(self):
        train_dataset = [1, 2, 3, 4]

        try:
            CrossFoldValidation(dataset=train_dataset, valid_size=1e-10)
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), CrossFoldValidation.INVALID_VALID_SIZE_MESSAGE)

    def test_can_not_split_when_valid_size_is_one(self):
        train_dataset = [1, 2, 3, 4]

        try:
            CrossFoldValidation(dataset=train_dataset, valid_size=1)
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), CrossFoldValidation.INVALID_VALID_SIZE_MESSAGE)

    def test_can_not_split_when_valid_size_is_all_the_dataset(self):
        train_dataset = [1, 2, 3, 4]

        try:
            CrossFoldValidation(dataset=train_dataset, valid_size=1-1e-10)
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), CrossFoldValidation.INVALID_VALID_SIZE_MESSAGE)

    def test_train_test_split(self):
        dataset = [1, 2, 3, 4]
        train_ds, val_ds = train_test_split(dataset=dataset, valid_size=0.5)
        self.assertDatasetEquals(train_ds, [3, 1])
        self.assertDatasetEquals(val_ds, [2, 4])

    def test_split_when_valid_size_is_a_half_returns_datasets_of_same_size(self):
        train_dataset = [1, 2, 3, 4]

        splitter = CrossFoldValidation(dataset=train_dataset, valid_size=0.5)
        it = iter(splitter)

        self.assertEqual(len(splitter), 2)

        train_dl, valid_dl = next(it)

        self.assertDatasetEquals(train_dl, [3, 1])
        self.assertDatasetEquals(valid_dl, [2, 4])

        train_dl, valid_dl = next(it)

        self.assertDatasetEquals(valid_dl, [3, 1])
        self.assertDatasetEquals(train_dl, [2, 4])

        self.assertStopIteration(it)

    def test_split_when_valid_size_is_not_multiple_of_dataset_varies_valid_dataset_size(self):
        train_dataset = [1, 2, 3, 4, 5]

        splitter = CrossFoldValidation(dataset=train_dataset, valid_size=0.4)
        it = iter(splitter)

        self.assertEqual(len(splitter), 3)

        train_dl, valid_dl = next(it)

        self.assertDatasetEquals(valid_dl, [1, 5])
        self.assertDatasetEquals(train_dl, [3, 4, 2])

        train_dl, valid_dl = next(it)

        self.assertDatasetEquals(valid_dl, [3, 4])
        self.assertDatasetEquals(train_dl, [1, 5, 2])

        train_dl, valid_dl = next(it)

        self.assertDatasetEquals(valid_dl, [2])
        self.assertDatasetEquals(train_dl, [1, 5, 3, 4])

        self.assertStopIteration(it)

    def test_train_and_validation_dataset_can_be_different(self):
        train_dataset = [1, 2, 3, 4, 5]
        valid_dataset = [6, 7, 8, 9, 10]

        splitter = CrossFoldValidation(dataset=train_dataset,
                                       valid_dataset=valid_dataset,
                                       valid_size=0.4)
        it = iter(splitter)

        self.assertEqual(len(splitter), 3)

        train_dl, valid_dl = next(it)

        self.assertDatasetEquals(valid_dl, [6, 10])
        self.assertDatasetEquals(train_dl, [3, 4, 2])

        train_dl, valid_dl = next(it)

        self.assertDatasetEquals(valid_dl, [8, 9])
        self.assertDatasetEquals(train_dl, [1, 5, 2])

        train_dl, valid_dl = next(it)

        self.assertDatasetEquals(valid_dl, [7])
        self.assertDatasetEquals(train_dl, [1, 5, 3, 4])

        self.assertStopIteration(it)

    def test_constructor_should_raise_if_train_and_valid_datasets_have_not_the_same_size(self):
        train_dataset = [1, 2]
        valid_dataset = [3]

        try:
            CrossFoldValidation(dataset=train_dataset,
                                valid_dataset=valid_dataset,
                                valid_size=0.5)
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), CrossFoldValidation.TRAIN_AND_VALID_DATASET_SIZE_MESSAGE)

    def test_unsupervise_dataset_hide_targets(self):
        supervised_dataset = [(1, True), (2, False), (3, False)]
        unsupervised_dataset = UnsuperviseDataset(supervised_dataset, input_indices=[0])

        self.assertEqual(list(unsupervised_dataset), [1, 2, 3])

    def test_unsupervise_dataset_can_select_over_input_indices(self):
        supervised_dataset = [(1, 3, True), (2, 4, False), (5, 6, False)]
        unsupervised_dataset = UnsuperviseDataset(supervised_dataset, input_indices=[0, 1])

        self.assertEqual(list(unsupervised_dataset), [(1, 3), (2, 4), (5, 6)])

    def test_shrink_to_p_0_is_equivalent_to_empty_dataset(self):
        dataset = list(range(1, 11))
        shrinked_dataset = ShrinkDataset(dataset, p=0)

        self.assertEqual(list(shrinked_dataset), [])

    def test_shrink_to_infimal_p_is_equivalent_to_empty_dataset(self):
        dataset = list(range(1, 11))
        shrinked_dataset = ShrinkDataset(dataset, p=0.001)

        self.assertEqual(list(shrinked_dataset), [])

    def test_shrink_to_p_1_is_all_dataset(self):
        dataset = list(range(1, 11))
        shrinked_dataset = ShrinkDataset(dataset, p=1)

        self.assertEqual(list(shrinked_dataset), dataset)

    def test_shrink_to_p_0p4_gets_gives_only_four_elements(self):
        dataset = list(range(1, 11))
        shrinked_dataset = ShrinkDataset(dataset, p=0.4)

        self.assertEqual(list(shrinked_dataset), [2, 3, 6, 7])

    def test_shrink_to_p_less_than_0_raises_exception(self):
        dataset = list(range(1, 11))
        try:
            shrinked_dataset = ShrinkDataset(dataset, p=-0.01)
            self.fail()
        except ValueError as e:
            self.assertEqual(str(e), ShrinkDataset.INVALID_P_MESSAGE)

    def test_shrink_to_p_greater_than_1_raises_exception(self):
        dataset = list(range(1, 11))
        try:
            shrinked_dataset = ShrinkDataset(dataset, p=1.01)
            self.fail()
        except ValueError as e:
            self.assertEqual(str(e), ShrinkDataset.INVALID_P_MESSAGE)

class TextClassificationBaseTest(object):
    def test_multilabel_dataset(self):
        ds = self.multilabel_ds
        self.assertEqual(len(ds), 2)
        x, y = ds[0]
        self.assertEqual(x, 'text b')
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.ndim, 1)
        self.assertEqual(y.tolist(), [0, 1])
        x, y = ds[1]
        self.assertEqual(x, 'text a')
        self.assertEqual(y.tolist(), [1, 0])

    def test_names(self):
        self.assertEqual(self.multilabel_ds.names, ['text'])
        ds = self.multiclass_ds
        self.assertEqual(self.multiclass_ds.names, ['text'])

    def test_show_samples(self):
        df_samples = self.multilabel_ds.show_samples(2)
        self.assertIsInstance(df_samples, pd.DataFrame)
        self.assertEqual(set(df_samples.itertuples(index=False, name=None)),
                         {('text a', 1, 0),
                          ('text b', 0, 1)})

    def test_multiclass_dataset(self):
        ds = self.multiclass_ds
        self.assertEqual(len(ds), 3)
        x, y = ds[0]
        self.assertEqual(x, 'text b')
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.ndim, 0)
        self.assertEqual(y.item(), 1)
        x, y = ds[1]
        self.assertEqual(x, 'text c')
        self.assertEqual(y.item(), 2)
        x, y = ds[2]
        self.assertEqual(x, 'text a')
        self.assertEqual(y.item(), 0)


class TextClassificationJSONDatasetTest(unittest.TestCase, TextClassificationBaseTest):
    def setUp(self):
        json_multilabel_dataset = [
            {'text': 'text b', 'class_a': 0, 'class_b': 1},
            {'text': 'text a', 'class_a': 1, 'class_b': 0}
        ]
        with tempfile.NamedTemporaryFile() as file:
            with open(file.name, 'w') as fp:
                json.dump(json_multilabel_dataset, fp)
            self.multilabel_ds = TextClassificationDataset.from_json(path=file.name,
                                                                     field_names='text',
                                                                     target_field_names=['class_a', 'class_b'])

        json_multiclass_dataset = [
            {'text': 'text b', 'class': 'b'},
            {'text': 'text c', 'class': 'c'},
            {'text': 'text a', 'class': 'a'},
        ]
        with tempfile.NamedTemporaryFile() as file:
            with open(file.name, 'w') as fp:
                json.dump(json_multiclass_dataset, fp)
            self.multiclass_ds = TextClassificationDataset.from_json(path=file.name,
                                                                     field_names='text',
                                                                     target_field_names='class')


class TextClassificationCSVDatasetTest(unittest.TestCase, TextClassificationBaseTest):
    def setUp(self):
        multilabel_dataset = ('text,class_a,class_b',
                              '\ntext b,0,1',
                              '\ntext a,1,0')
        with tempfile.NamedTemporaryFile() as file:
            with open(file.name, 'w') as fp:
                fp.writelines(multilabel_dataset)
            self.multilabel_ds = TextClassificationDataset.from_csv(path=file.name,
                                                     columns='text',
                                                     target_columns=['class_a', 'class_b'])

        multiclass_dataset = (
            'text,class',
            '\ntext b,b',
            '\ntext c,c',
            '\ntext a,a',
        )
        with tempfile.NamedTemporaryFile() as file:
            with open(file.name, 'w') as fp:
                fp.writelines(multiclass_dataset)
            self.multiclass_ds = TextClassificationDataset.from_csv(path=file.name,
                                                                     columns='text',
                                                                     target_columns='class')

class TextClassificationDFDatasetTest(unittest.TestCase, TextClassificationBaseTest):
    def setUp(self):
        df_multilabel = pd.DataFrame({
            'text': ['text b', 'text a'],
            'class_a': [0, 1],
            'class_b': [1, 0]
        })
        self.multilabel_ds = TextClassificationDataset(df_multilabel['text'],
                                                       df_multilabel[['class_a', 'class_b']])

        df_multiclass = pd.DataFrame({
            'text': ['text b', 'text c', 'text a'],
            'class': ['b', 'c', 'a'],
        })
        self.multiclass_ds = TextClassificationDataset(df_multiclass[['text']],
                                                       df_multiclass[['class']])