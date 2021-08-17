import shutil
from pathlib import Path

from .common import *
from torchero.utils.format import format_metric
from torchero.utils.collate import PadSequenceCollate, BoWCollate
from torchero.utils import download_from_url
from torchero.utils.vision import download_image, download_images

from PIL import Image

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


class IOTests(unittest.TestCase):
    def test_download_from_url(self):
        url = 'https://raw.githubusercontent.com/juancruzsosa/torchero/master/tests/__init__.py'
        dst = '/tmp/download.py'
        download_from_url(url, dst, pbar=True)
        with open(dst, 'rb') as fp:
            remote_content = fp.read()
        with open(Path(__file__).parent/'__init__.py', 'rb') as fp:
            local_content = fp.read()
        self.assertEqual(remote_content, local_content)

    def test_download_image_from_url(self):
        url = 'https://raw.githubusercontent.com/juancruzsosa/torchero/master/documentation/source/img/quickstart/confusion_matrix.png'
        src = Path(__file__).parent.parent/'documentation'/'source'/'img'/'quickstart'/'confusion_matrix.png'
        remote_im = download_image(url)
        local_im = Image.open(src)
        self.assertEqual(remote_im, local_im)

    def test_download_multiple_images(self):
        urls = ['https://raw.githubusercontent.com/juancruzsosa/torchero/master/documentation/source/img/quickstart/confusion_matrix.png',
                'https://raw.githubusercontent.com/juancruzsosa/torchero/master/documentation/source/img/quickstart/metrics.png']
        root = Path(__file__).parent.parent/'documentation'/'source'/'img'/'quickstart'
        srcs = [root/'confusion_matrix.png',
                root/'metrics.png']
        for remote_im, src in zip(download_images(urls, pbar=True, num_workers=2), srcs):
            local_im = Image.open(src)
            self.assertEqual(remote_im, local_im)

class CollateTests(unittest.TestCase):
    def setUp(self):
        self.batch = [(torch.tensor([1, 2, 3, 4]),    torch.tensor(1.0)),
                      (torch.tensor([5]),             torch.tensor(0.0)),
                      (torch.tensor([6, 7, 8]),       torch.tensor(1.0)),
                      (torch.tensor([9, 10]),         torch.tensor(0.0)),
                      (torch.tensor([11, 12, 13, 14]),torch.tensor(1.0))]
    def assertTensorsEqual(self, a, b):
        self.assertTrue(torch.is_tensor(a))
        self.assertTrue(torch.is_tensor(b))

        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(a.tolist(), b.tolist())

    def test_pad_sequence_collate_one_sequence(self):
        collate =  PadSequenceCollate()
        X, y = torch.tensor([1, 2, 3]), torch.tensor(1.0)
        (X_padded, lengths), y = collate([(X, y)])
        self.assertTensorsEqual(X_padded, torch.tensor([[1, 2, 3]]))
        self.assertTensorsEqual(lengths, torch.LongTensor([3]))
        self.assertTensorsEqual(y, torch.tensor([1.0]))

    def test_left_padding(self):
        collate =  PadSequenceCollate(padding_scheme='left')
        (X_padded, lengths), y = collate(self.batch)
        self.assertTensorsEqual(X_padded, torch.tensor([[1,   2,  3,   4],
                                                        [0,   0,  0,   5],
                                                        [0,   6,  7,   8],
                                                        [0,   0,  9,  10],
                                                        [11, 12, 13, 14]]))
        self.assertTensorsEqual(lengths, torch.LongTensor([4, 1, 3, 2, 4]))
        self.assertTensorsEqual(y, torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]))

    def test_right_padding_with_different_pad_value(self):
        collate =  PadSequenceCollate(padding_scheme='right', pad_value=-1)
        (X_padded, lengths), y = collate(self.batch)
        self.assertTensorsEqual(X_padded, torch.tensor([[1,   2,  3,   4],
                                                        [5,  -1, -1,  -1],
                                                        [6,  7,  8,   -1],
                                                        [9,   10, -1, -1],
                                                        [11, 12, 13, 14]]))
        self.assertTensorsEqual(lengths, torch.LongTensor([4, 1, 3, 2, 4]))
        self.assertTensorsEqual(y, torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]))

    def test_center_padding(self):
        collate =  PadSequenceCollate(padding_scheme='center')
        (X_padded, lengths), y = collate(self.batch)
        self.assertTensorsEqual(X_padded, torch.tensor([[1,   2,  3,   4],
                                                        [0,   0,  5,   0],
                                                        [0,   6,  7,   8],
                                                        [0,   9, 10,   0],
                                                        [11, 12, 13, 14]]))
        self.assertTensorsEqual(lengths, torch.LongTensor([4, 1, 3, 2, 4]))
        self.assertTensorsEqual(y, torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]))

    def test_bow_padding(self):
        collate =  BoWCollate()
        (X_padded, offsets), y = collate(self.batch)
        self.assertTensorsEqual(X_padded, torch.arange(1, 15).long())
        self.assertTensorsEqual(offsets, torch.LongTensor([0, 4, 5, 8, 10]))
        self.assertTensorsEqual(y, torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]))

if __name__ == '__main__':
    unittest.main()
