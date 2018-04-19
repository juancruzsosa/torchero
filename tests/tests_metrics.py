import torch
from torch import nn
import unittest
import torchtrainer
import math
from torchtrainer import meters
from torchtrainer.meters.aggregators import batch, scale

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

        meter_normalized = meters.CategoricalAccuracy(aggregator=batch.Average())
        meter_sum = meters.CategoricalAccuracy(aggregator=batch.Sum())
        meter_percentage = meters.CategoricalAccuracy(aggregator=scale.percentage(batch.Average()))

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

        meter_normalized = meters.CategoricalAccuracy(aggregator=batch.Average())
        meter_sum = meters.CategoricalAccuracy(aggregator=batch.Sum())
        meter_percentage = meters.CategoricalAccuracy(aggregator=scale.percentage(batch.Average()))

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

    def test_binary_meters_with_incresing_threholds(self):
        a1 = torch.Tensor([[0.3]])
        a2 = torch.Tensor([[0.5]])
        a3 = torch.Tensor([[0.7]])

        t1 = torch.LongTensor([1])
        t2 = torch.LongTensor([0])

        meter_th_p2 = meters.BinaryAccuracy(aggregator=batch.Average(), threshold=0.2)
        meter_th_p5 = meters.BinaryAccuracy(aggregator=batch.Average(), threshold=0.5)
        meter_th_p8 = meters.BinaryAccuracy(aggregator=batch.Average(), threshold=0.8)

        self.assertMeasureEqual(meter_th_p2, [(a1, t1)], 1.0)
        self.assertMeasureEqual(meter_th_p5, [(a1, t1)], 0.0)
        self.assertMeasureEqual(meter_th_p8, [(a1, t1)], 0.0)

        self.assertMeasureEqual(meter_th_p2, [(a2, t1)], 1.0)
        self.assertMeasureEqual(meter_th_p5, [(a2, t1)], 1.0)
        self.assertMeasureEqual(meter_th_p8, [(a2, t1)], 0.0)

        self.assertMeasureEqual(meter_th_p2, [(a3, t2)], 0.0)
        self.assertMeasureEqual(meter_th_p5, [(a3, t2)], 0.0)
        self.assertMeasureEqual(meter_th_p8, [(a3, t2)], 1.0)

    def test_binary_accuracy_with_multiples_batches(self):
        a = torch.Tensor([[0.3],
                          [0.5],
                          [0.7]])

        t = torch.LongTensor([1, 0, 0])

        meter_th_p2 = meters.BinaryAccuracy(aggregator=batch.Average(), threshold=0.2)
        meter_th_p5 = meters.BinaryAccuracy(aggregator=batch.Average(), threshold=0.5)
        meter_th_p8 = meters.BinaryAccuracy(aggregator=batch.Average(), threshold=0.8)

        self.assertMeasureAlmostEqual(meter_th_p2, [(a, t)], 1/3)
        self.assertMeasureAlmostEqual(meter_th_p5, [(a, t)], 0.0)
        self.assertMeasureAlmostEqual(meter_th_p8, [(a, t)], 2/3)


    def test_binary_accyracy_with_logits(self):
        a1 = torch.Tensor([[-1]])
        a2 = torch.Tensor([[0]])
        a3 = torch.Tensor([[1]])

        t1 = torch.LongTensor([1])
        t2 = torch.LongTensor([0])

        meter_th_p2 = meters.BinaryWithLogitsAccuracy(aggregator=batch.Average(), threshold=0.2)
        meter_th_p5 = meters.BinaryWithLogitsAccuracy(aggregator=batch.Average(), threshold=0.5)
        meter_th_p8 = meters.BinaryWithLogitsAccuracy(aggregator=batch.Average(), threshold=0.8)

        self.assertMeasureEqual(meter_th_p2, [(a1, t1)], 1.0)
        self.assertMeasureEqual(meter_th_p5, [(a1, t1)], 0.0)
        self.assertMeasureEqual(meter_th_p8, [(a1, t1)], 0.0)

        self.assertMeasureEqual(meter_th_p2, [(a2, t1)], 1.0)
        self.assertMeasureEqual(meter_th_p5, [(a2, t1)], 1.0)
        self.assertMeasureEqual(meter_th_p8, [(a2, t1)], 0.0)

        self.assertMeasureEqual(meter_th_p2, [(a3, t2)], 0.0)
        self.assertMeasureEqual(meter_th_p5, [(a3, t2)], 0.0)
        self.assertMeasureEqual(meter_th_p8, [(a3, t2)], 1.0)

    def test_binary_accyracy_with_custom_activation(self):
        a1 = torch.Tensor([[-1]])
        a2 = torch.Tensor([[0]])
        a3 = torch.Tensor([[1]])

        t1 = torch.LongTensor([1])
        t2 = torch.LongTensor([0])

        meter_th_p2 = meters.BinaryWithLogitsAccuracy(aggregator=batch.Average(), threshold=-0.8, activation=nn.Tanh())
        meter_th_p5 = meters.BinaryWithLogitsAccuracy(aggregator=batch.Average(), threshold=0, activation=nn.Tanh())
        meter_th_p8 = meters.BinaryWithLogitsAccuracy(aggregator=batch.Average(), threshold=0.8, activation=nn.Tanh())

        self.assertMeasureEqual(meter_th_p2, [(a1, t1)], 1.0)
        self.assertMeasureEqual(meter_th_p5, [(a1, t1)], 0.0)
        self.assertMeasureEqual(meter_th_p8, [(a1, t1)], 0.0)

        self.assertMeasureEqual(meter_th_p2, [(a2, t1)], 1.0)
        self.assertMeasureEqual(meter_th_p5, [(a2, t1)], 1.0)
        self.assertMeasureEqual(meter_th_p8, [(a2, t1)], 0.0)

        self.assertMeasureEqual(meter_th_p2, [(a3, t2)], 0.0)
        self.assertMeasureEqual(meter_th_p5, [(a3, t2)], 0.0)
        self.assertMeasureEqual(meter_th_p8, [(a3, t2)], 1.0)


class MSETests(BaseMetricsTests):
    def test_meter_measure_is_always_positive(self):
        meter = meters.MSE()
        sqrt_meter = meters.MSE(take_sqrt=True)
        self.assertMeasureEqual(meter, [(torch.ones(1,1), torch.ones(1,1))], 0)
        self.assertMeasureEqual(meter, [(torch.ones(1,1), torch.zeros(1,1))], 1)
        self.assertMeasureEqual(meter, [(torch.zeros(1,1), torch.ones(1,1))], 1)
        self.assertMeasureEqual(meter, [(-torch.ones(1,2), torch.zeros(1,2))], 2*1)
        self.assertMeasureEqual(meter, [(torch.zeros(1,2), -torch.ones(1,2))], 2*1)
        self.assertMeasureEqual(meter, [(2*torch.ones(1,2), torch.zeros(1,2))], 4*2)
        self.assertMeasureEqual(meter, [(torch.zeros(1,2), 2*torch.ones(1,2))], 4*2)
        self.assertMeasureEqual(meter, [(-2*torch.ones(1,2), torch.zeros(1,2))], 4*2)
        self.assertMeasureEqual(meter, [(torch.zeros(1,2), -2*torch.ones(1,2))], 4*2)
        self.assertMeasureAlmostEqual(sqrt_meter, [(torch.zeros(1,2), -2*torch.ones(1,2))], math.sqrt(4*2))

    def test_cannot_measure_with_1d_tensors(self):
        a = torch.Tensor([0.2])
        t = torch.Tensor([0.1])
        meter = meters.MSE()

        try:
            meter.measure(a,t)
            self.fail()
        except ValueError as e:
            self.assertEqual(str(e), meter.INVALID_BATCH_DIMENSION_MESSAGE)

    def test_cannot_measure_with_different_shape_tensors(self):
        a = torch.Tensor([[0.2, 0.1]])
        t = torch.Tensor([[0.1]])
        meter = meters.MSE()

        try:
            meter.measure(a,t)
            self.fail()
        except ValueError as e:
            self.assertEqual(str(e), meter.INVALID_BATCH_DIMENSION_MESSAGE)

    def test_meter_value_average_over_batch_dimention(self):
        meter = meters.MSE()
        sqrt_meter = meters.MSE(take_sqrt=True)
        self.assertMeasureEqual(meter, [(torch.ones(2,1), torch.zeros(2,1))], 1)
        self.assertMeasureEqual(meter, [(torch.zeros(2,1), torch.ones(2,1))], 1)
        self.assertMeasureEqual(meter, [(2*torch.ones(2,1), torch.zeros(2,1))], 4)
        self.assertMeasureEqual(meter, [(torch.zeros(2,1), 2*torch.ones(2,1))], 4)
        self.assertMeasureEqual(meter, [(torch.zeros(2,1), 2*torch.ones(2,1))], 4)
        self.assertMeasureAlmostEqual(meter, [(torch.arange(0, 3).view(3, 1), torch.arange(3, 6).view(3, 1))], 3**2)
        self.assertMeasureAlmostEqual(sqrt_meter, [(torch.arange(0, 3).view(3, 1), torch.arange(3, 6).view(3, 1))], 3)

    def test_meter_value_average_over_sum_of_measured_batch_dimentions(self):
        meter = meters.MSE()
        sqrt_meter = meters.MSE(take_sqrt=True)
        self.assertMeasureAlmostEqual(meter, [(torch.ones(2,1), torch.zeros(2,1)),
                                              (2*torch.ones(2,1), torch.zeros(2,1)),
                                              (torch.arange(0, 3).view(3, 1), torch.arange(3, 6).view(3, 1))], (2*1**2 + 2*2**2 + 3*3**2)/7)
        self.assertMeasureAlmostEqual(sqrt_meter, [(torch.ones(2,1), torch.zeros(2,1)),
                                              (2*torch.ones(2,1), torch.zeros(2,1)),
                                              (torch.arange(0, 3).view(3, 1), torch.arange(3, 6).view(3, 1))], math.sqrt((2*1**2 + 2*2**2 + 3*3**2)/7))

    def test_cannot_measure_with_different_type_of_tensors(self):
        from torch.autograd import Variable
        import numpy as np

        a = [[0.2]]
        meter = meters.MSE()

        try:
            meter.measure(np.array(a), np.array(a))
            self.fail()
        except TypeError as e:
            self.assertEqual(str(e), meter.INVALID_INPUT_TYPE_MESSAGE)

        try:
            meter.measure(a, a)
            self.fail()
        except TypeError as e:
            self.assertEqual(str(e), meter.INVALID_INPUT_TYPE_MESSAGE)

        try:
            meter.measure(Variable(torch.Tensor(a)), Variable(torch.Tensor(a)))
            self.fail()
        except TypeError as e:
            self.assertEqual(str(e), meter.INVALID_INPUT_TYPE_MESSAGE)

    def test_cannot_get_value_with_no_measures(self):
        meter = meters.MSE()
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
