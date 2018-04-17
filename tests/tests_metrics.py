import torch
import unittest
import torchtrainer
from torchtrainer import meters
from torchtrainer.meters import ResultMode

class BaseMetricsTests(unittest.TestCase):
    def measure_once(self, meter, batchs):
        meter.reset()
        for x_batch, y_batch in batchs:
            meter.measure(x_batch, y_batch)
        return meter.value()

    def assertMeasureEqual(self, meter, batchs, measure):
        self.assertEqual(self.measure_once(meter, batchs), measure)

    def assertMeasureAlmostEqual(self, meter, batchs, measure):
        self.assertAlmostEqual(self.measure_once(meter, batchs), measure)

class AccuracyMetricsTests(BaseMetricsTests):
    def test_classification_metter_only_checks_indices_of_maximum_value(self):
        a = torch.Tensor([[1]])
        t = torch.LongTensor([0])

        a1 = torch.Tensor([[0.5, 0.3, 0.2]])
        t1 = torch.LongTensor([0])

        a2 = torch.Tensor([[-1.0, 2.0,  1.0]])
        t2 = torch.LongTensor([1])

        a3 = torch.Tensor([[1.0, 2.0, 3.0]])
        t3 = torch.LongTensor([2])

        meter = meters.CategoricalAccuracy()
        self.assertMeasureEqual(meter, [(a, t)], 1.0)

        self.assertMeasureEqual(meter, [(a1, t1)], 1.0)
        self.assertMeasureEqual(meter, [(a1, t2)], 0.0)
        self.assertMeasureEqual(meter, [(a1, t3)], 0.0)

        self.assertMeasureEqual(meter, [(a2, t1)], 0.0)
        self.assertMeasureEqual(meter, [(a2, t2)], 1.0)
        self.assertMeasureEqual(meter, [(a2, t3)], 0.0)

        self.assertMeasureEqual(meter, [(a3, t1)], 0.0)
        self.assertMeasureEqual(meter, [(a3, t2)], 0.0)
        self.assertMeasureEqual(meter, [(a3, t3)], 1.0)

    def test_size_average_option_average_results_over_the_batch_dimension(self):
        a = torch.Tensor([[0.55, 0.45],
                          [-1.0, 2.0]])
        t1 = torch.LongTensor([0, 0])
        t2 = torch.LongTensor([0, 1])
        t3 = torch.LongTensor([1, 0])
        t4 = torch.LongTensor([1, 1])

        meter_normalized = meters.CategoricalAccuracy(result_mode=ResultMode.NORMALIZED)
        meter_sum = meters.CategoricalAccuracy(result_mode=ResultMode.SUM)
        meter_percentage = meters.CategoricalAccuracy(result_mode=ResultMode.PERCENTAGE)

        self.assertMeasureAlmostEqual(meter_normalized, [(a, t1)], 1/2)
        self.assertMeasureAlmostEqual(meter_normalized, [(a, t2)], 1)
        self.assertMeasureAlmostEqual(meter_normalized, [(a, t3)], 0)
        self.assertMeasureAlmostEqual(meter_normalized, [(a, t4)], 1/2)

        self.assertMeasureAlmostEqual(meter_sum, [(a, t1)], 1)
        self.assertMeasureAlmostEqual(meter_sum, [(a, t2)], 2)
        self.assertMeasureAlmostEqual(meter_sum, [(a, t3)], 0)
        self.assertMeasureAlmostEqual(meter_sum, [(a, t4)], 1)

        self.assertMeasureAlmostEqual(meter_percentage, [(a, t1)], 50.0)
        self.assertMeasureAlmostEqual(meter_percentage, [(a, t2)], 100.0)
        self.assertMeasureAlmostEqual(meter_percentage, [(a, t3)], 0.0)
        self.assertMeasureAlmostEqual(meter_percentage, [(a, t4)], 50.0)

    def test_cannot_measure_with_1d_tensors(self):
        a = torch.Tensor([0.1])
        t = torch.LongTensor([0])
        meter = meters.CategoricalAccuracy()

        try:
            meter.measure(a,t)
            self.fail()
        except ValueError as e:
            self.assertEqual(str(e), meter.INVALID_BATCH_DIMENSION_MESSAGE)

    def test_cannot_measure_with_different_number_of_classes(self):
        a = torch.Tensor([[0.1]])
        b = torch.LongTensor([0, 0])
        meter = meters.CategoricalAccuracy()

        try:
            meter.measure(a, b)
            self.fail()
        except ValueError as e:
            self.assertEqual(str(e), meter.INVALID_BATCH_DIMENSION_MESSAGE)

    def test_cannot_measure_inputs_other_than_tensors(self):
        from torch.autograd import Variable
        a = torch.Tensor([[0.1]])
        va = Variable(a)
        t = torch.FloatTensor([0])
        lt = torch.LongTensor([0])
        vt = Variable(lt)
        meter = meters.CategoricalAccuracy()

        try:
            meter.measure(va, lt)
            self.fail()
        except TypeError as e:
            self.assertEqual(str(e), meter.INVALID_INPUT_TYPE_MESSAGE)

        try:
            meter.measure(a, t)
            self.fail()
        except TypeError as e:
            self.assertEqual(str(e), meter.INVALID_INPUT_TYPE_MESSAGE)

        try:
            meter.measure(a, vt)
            self.fail()
        except TypeError as e:
            self.assertEqual(str(e), meter.INVALID_INPUT_TYPE_MESSAGE)

    def test_size_average_option_average_results_over_the_batch_dimension_on_multiples_passes(self):
        a1 = torch.Tensor([[0.5, 0.3, 0.2],
                          [-1.0, 2.0,  1.0]])
        a2 = torch.Tensor([[1.0, 2.0, 3.0]])

        t1 = torch.LongTensor([0, 0])
        t2 = torch.LongTensor([2])

        meter_normalized = meters.CategoricalAccuracy(result_mode=ResultMode.NORMALIZED)
        meter_sum = meters.CategoricalAccuracy(result_mode=ResultMode.SUM)
        meter_percentage = meters.CategoricalAccuracy(result_mode=ResultMode.PERCENTAGE)

        self.assertMeasureAlmostEqual(meter_normalized, [(a1, t1), (a2, t2)], 2/3)

        self.assertMeasureAlmostEqual(meter_sum, [(a1, t1), (a2, t2)], 2)

        self.assertMeasureAlmostEqual(meter_percentage, [(a1, t1), (a2, t2)], 2*100/3)

    def test_cannot_get_value_with_no_measures(self):
        meter = meters.CategoricalAccuracy()
        try:
            meter.value()
            self.fail()
        except meters.ZeroMeasurementsError as e:
            pass

class AveragerTests(BaseMetricsTests):
    def measure_once(self, meter, xs):
        meter.reset()
        for x in xs:
            meter.measure(x)
        return meter.value()

    def test_averager_cannot_return_results_with_no_measures(self):
        meter = meters.Averager()
        try:
            meter.value()
            self.fail()
        except meters.ZeroMeasurementsError as e:
            pass

    def test_averager_value_is_the_average_over_all_measures(self):
        meter = meters.Averager()
        self.assertMeasureEqual(meter, [1], 1)
        self.assertMeasureAlmostEqual(meter, [1, 2], (1+2)/2)
        self.assertMeasureAlmostEqual(meter, [1, 2, 3], (1+2+3)/3)
