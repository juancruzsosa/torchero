import shutil
from .common import *
from torchero.utils.format import format_metric

class FormatMetricTest(unittest.TestCase):
    def test_medium_numbers_get_truncated(self):
        self.assertEqual(format_metric(-0.5), '-0.500')
        self.assertEqual(format_metric(0.5), '0.500')
        self.assertEqual(format_metric(0.02), '0.020')
        self.assertEqual(format_metric(0.002), '0.002')
        self.assertEqual(format_metric(0.5449), '0.545')

    def test_very_small_numbers_get_formatted_with_negative_(self):
        self.assertEqual(format_metric(0.0000032), '3.20e-06')

    def test_format_string(self):
        self.assertEqual(format_metric(''), '')
        self.assertEqual(format_metric('example'), 'example')

    def test_0d_tensor_get_formatted_as_floats(self):
        self.assertEqual(format_metric(torch.tensor(1)), '1')
        self.assertEqual(format_metric(torch.tensor(2, dtype=torch.int32)), '2')
        self.assertEqual(format_metric(torch.tensor(0.5)), '0.5')

    def test_1d_tensor_get_formatted_as_lists(self):
        self.assertEqual(format_metric(torch.tensor([1, 2, 3])), '[1, 2, 3]')

    def test_2d_tensor_get_formatted_as_nested_lists(self):
        self.assertEqual(format_metric(torch.tensor([[1, 0], [2, 1], [3, 2]])), '[[1, 0], [2, 1], [3, 2]]')

    def test_list_get_formatted_as_lists(self):
        self.assertEqual(format_metric([2, 3]), "[2, 3]")
        self.assertEqual(format_metric([torch.tensor(2), torch.tensor(3)]), "[2, 3]")

    def test_tuple_get_formatted_as_lists(self):
        self.assertEqual(format_metric((2, 3)), "[2, 3]")
        self.assertEqual(format_metric((torch.tensor(2), torch.tensor(3))), "[2, 3]")

    def test_format_dict(self):
        self.assertIn(format_metric({'a': 1, 'b': 2}), ["{'a': 1, 'b': 2}", "{'b': 2, 'a': 1}"])
        self.assertIn(format_metric({'a': torch.tensor(1), 'b': torch.tensor(2)}), ["{'a': 1, 'b': 2}", "{'b': 2, 'a': 1}"])
