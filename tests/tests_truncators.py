import unittest
from torchero.utils.text.transforms import LeftTruncator, RightTruncator, CenterTruncator

class TruncatorTests(unittest.TestCase):
    def setUp(self):
        self.short_text = ['Very', 'short', 'text']
        self.long_text = ['This', 'is', 'a', 'very', 'long', 'text', 'to', 'test']

    def tests_left_truncate(self):
        truncator = LeftTruncator(4)
        self.assertEqual(truncator(self.short_text), ['Very', 'short', 'text'])
        self.assertEqual(truncator(self.long_text), ['This', 'is', 'a', 'very'])

    def tests_right_truncate(self):
        truncator = RightTruncator(4)
        self.assertEqual(truncator(self.short_text), ['Very', 'short', 'text'])
        self.assertEqual(truncator(self.long_text), ['long', 'text', 'to', 'test'])

    def tests_center_truncate(self):
        truncator = CenterTruncator(4)
        self.assertEqual(truncator(self.short_text), ['Very', 'short', 'text'])
        self.assertEqual(truncator(self.long_text), ['a', 'very', 'long', 'text'])
