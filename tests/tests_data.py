import torch
import torchtrainer
from torchtrainer.utils.data import CrossFoldValidation
import unittest

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
