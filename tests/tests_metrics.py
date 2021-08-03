import math
from .common import *
from torchero.meters.aggregators import batch, scale
from torchero.utils.defaults import parse_meters
from time import sleep

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

        for i, a in enumerate([a1, a2, a3]):
            for j, t in enumerate([t1, t2, t3]):
                self.assertMeasureEqual(meter, [(a, t)], 1.0 if i == j else 0.0)

    def test_classification_with_k_greater_than_one_search_top_k_indices(self):
        a1 = torch.Tensor([[0.5, 0.4, 0.1]])
        a2 = torch.Tensor([[20, 5, 10]])
        a3 = torch.Tensor([[2, 3, 5]])

        t1 = torch.LongTensor([2])
        t2 = torch.LongTensor([1])
        t3 = torch.LongTensor([0])

        meter = meters.CategoricalAccuracy(k=2)

        for i, a in enumerate([a1, a2, a3]):
            for j, t in enumerate([t1, t2, t3]):
                self.assertMeasureEqual(meter, [(a, t)], 0.0 if j == i else 1.0)


    def test_aggregators_works_over_the_batch_dimension(self):
        a = torch.Tensor([[0.55, 0.45],
                          [-1.0, 2.0]])
        t1 = torch.LongTensor([0, 0])
        t2 = torch.LongTensor([0, 1])
        t3 = torch.LongTensor([1, 0])
        t4 = torch.LongTensor([1, 1])

        meter_normalized = meters.CategoricalAccuracy(aggregator=batch.Average())
        meter_sum = meters.CategoricalAccuracy(aggregator=batch.Sum())
        meter_percentage = meters.CategoricalAccuracy(aggregator=scale.percentage(batch.Average()))
        meter_maximum = meters.CategoricalAccuracy(aggregator=batch.Maximum())
        meter_minimum = meters.CategoricalAccuracy(aggregator=batch.Minimum())

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

        self.assertMeasureAlmostEqual(meter_maximum, [(a, t1)], 1)
        self.assertMeasureAlmostEqual(meter_maximum, [(a, t2)], 1)
        self.assertMeasureAlmostEqual(meter_maximum, [(a, t3)], 0)
        self.assertMeasureAlmostEqual(meter_maximum, [(a, t4)], 1)

        self.assertMeasureAlmostEqual(meter_minimum, [(a, t1)], 0)
        self.assertMeasureAlmostEqual(meter_minimum, [(a, t2)], 1)
        self.assertMeasureAlmostEqual(meter_minimum, [(a, t3)], 0)
        self.assertMeasureAlmostEqual(meter_minimum, [(a, t4)], 0)

    def test_cannot_measure_with_other_type_than_tensors(self):
        meter = meters.CategoricalAccuracy()

        try:
            meter.measure([0.9, 0.1], [1])
            self.fail()
        except TypeError as e:
            self.assertEqual(str(e), meter.INVALID_INPUT_TYPE_MESSAGE)

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
        a = torch.Tensor([[0.1]])
        t = torch.FloatTensor([0])
        meter = meters.CategoricalAccuracy()

        try:
            meter.measure(a, t)
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
        self.assertMeasureAlmostEqual(meter_normalized, [(a1, t1), (a2, t2)], 2/3)

    def test_cannot_get_value_with_no_measures(self):
        meter = meters.CategoricalAccuracy()
        try:
            meter.value()
            self.fail()
        except meters.ZeroMeasurementsError as e:
            pass

    def test_binary_meters_with_incresing_threholds_change_region_decision(self):
        a1 = torch.Tensor([0.3])
        a2 = torch.Tensor([0.5])
        a3 = torch.Tensor([0.7])

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


    def test_binary_accuracy_with_logits_applies_activation_applies_activation_before_regiion_decision(self):
        a1 = torch.Tensor([-1])
        a2 = torch.Tensor([0])
        a3 = torch.Tensor([1])

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

    def test_binary_accuracy_with_custom_activation_applies_that_activation(self):
        a1 = torch.Tensor([-1])
        a2 = torch.Tensor([0])
        a3 = torch.Tensor([1])

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
    def test_meter_measure_is_the_square_of_the_difference(self):
        meter = meters.MSE()
        sqrt_meter = meters.RMSE()
        self.assertMeasureEqual(meter, [(torch.ones(1,1), torch.ones(1,1))], 0)
        self.assertMeasureEqual(meter, [(torch.ones(1,1), torch.zeros(1,1))], 1)
        self.assertMeasureEqual(meter, [(torch.zeros(1,1), torch.ones(1,1))], 1)
        self.assertMeasureEqual(meter, [(-torch.ones(1,2), torch.zeros(1,2))], 2/2)
        self.assertMeasureEqual(meter, [(torch.zeros(1,2), -torch.ones(1,2))], 2/2)
        self.assertMeasureEqual(meter, [(2*torch.ones(1,2), torch.zeros(1,2))], 2**2*2/2)
        self.assertMeasureEqual(meter, [(torch.zeros(1,2), 2*torch.ones(1,2))], 2**2*2/2)
        self.assertMeasureEqual(meter, [(-2*torch.ones(1,2), torch.zeros(1,2))], 2**2*2/2)
        self.assertMeasureEqual(meter, [(torch.zeros(1,2), -2*torch.ones(1,2))], 2**2*2/2)
        self.assertMeasureAlmostEqual(sqrt_meter, [(torch.zeros(1,2), -2*torch.ones(1,2))], math.sqrt(2**2*2/2))

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
        sqrt_meter = meters.RMSE()
        msle_meter = meters.MSLE()
        rmsle_meter = meters.RMSLE()
        self.assertMeasureEqual(meter, [(torch.ones(2,1), torch.zeros(2,1))], 1)
        self.assertMeasureEqual(meter, [(torch.zeros(2,1), torch.ones(2,1))], 1)
        self.assertMeasureEqual(meter, [(2*torch.ones(2,1), torch.zeros(2,1))], 4)
        self.assertMeasureEqual(meter, [(torch.zeros(2,1), 2*torch.ones(2,1))], 4)
        self.assertMeasureEqual(meter, [(torch.zeros(2,1), 2*torch.ones(2,1))], 4)
        self.assertMeasureAlmostEqual(meter, [(torch.arange(0, 3).float().view(3, 1), torch.arange(3, 6).float().view(3, 1))], 3**2)
        self.assertMeasureAlmostEqual(sqrt_meter, [(torch.arange(0, 3).float().view(3, 1), torch.arange(3, 6).float().view(3, 1))], 3)
        self.assertMeasureAlmostEqual(msle_meter, [(torch.Tensor([[math.exp(2)-1, math.exp(1)-1]]),
                                                    torch.Tensor([[math.exp(4)-1, math.exp(2)-1]]))],
                                      ((2-4)**2 + (1-2)**2) / 2)
        self.assertMeasureAlmostEqual(rmsle_meter, [(torch.Tensor([[math.exp(2)-1, math.exp(1)-1]]),
                                                     torch.Tensor([[math.exp(4)-1, math.exp(2)-1]]))],
                                      math.sqrt(((2-4)**2 + (1-2)**2)/2))

    def test_meter_value_average_over_sum_of_measured_batch_dimentions(self):
        meter = meters.MSE()
        sqrt_meter = meters.RMSE()
        self.assertMeasureAlmostEqual(meter, [(torch.ones(2,1), torch.zeros(2,1)),
                                              (2*torch.ones(2,1), torch.zeros(2,1))],
                                      (2*1**2 + 2*2**2)/4)
        self.assertMeasureAlmostEqual(sqrt_meter, [(torch.ones(2,1), torch.zeros(2,1)),
                                                   (2*torch.ones(2,1), torch.zeros(2,1))],
                                      math.sqrt((2*1**2 + 2*2**2)/4))

    def test_cannot_measure_with_different_type_of_tensors(self):
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

    def test_cannot_get_value_with_no_measures(self):
        meter = meters.MSE()
        try:
            meter.value()
            self.fail()
        except meters.ZeroMeasurementsError as e:
            pass

class ConfusionMatrixTests(BaseMetricsTests):
    def assertTensorsEqual(self, a, b):
        return self.assertEqual(a.tolist(), b.tolist())

    def assertMeasureEqual(self, meter, batchs, measure):
        self.assertTensorsEqual(self.measure_once(meter, batchs).matrix, measure)

    def test_can_not_create_meter_with_negative_classes(self):
        try:
            meters.ConfusionMatrix(nr_classes=0)
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), meters.ConfusionMatrix.INVALID_NR_OF_CLASSES_MESSAGE.format(nr_classes=0))

    def test_cannot_measure_anything_other_than_longtensor_or_byte_tensor_on_left_param(self):
        meter = meters.ConfusionMatrix(nr_classes=1)
        a = torch.zeros(1).type(torch.FloatTensor)
        b = torch.zeros(1).type(torch.LongTensor)
        try:
            meter.measure(a, b)
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), meter.INVALID_INPUT_TYPE_MESSAGE.format(type_=a.type()))

    def test_cannot_measure_anything_other_than_longtensor_or_byte_tensor_on_right_param(self):
        meter = meters.ConfusionMatrix(nr_classes=1)
        a = torch.zeros(1).type(torch.FloatTensor)
        b = torch.zeros(1).type(torch.LongTensor)
        try:
            meter.measure(b, a)
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), meter.INVALID_INPUT_TYPE_MESSAGE.format(type_=a.type()))

    # def test_cannot_measure_anything_other_1d_tensors_on_left_param(self):
    #     meter = meters.ConfusionMatrix(nr_classes=1)
    #     try:
    #         meter.measure(torch.LongTensor([[0]]), torch.LongTensor([0]))
    #         self.fail()
    #     except Exception as e:
    #         self.assertEqual(str(e), meter.INVALID_BATCH_DIMENSION_MESSAGE.format(dims=2))

    def test_cannot_measure_anything_other_1d_tensors_on_right_param(self):
        meter = meters.ConfusionMatrix(nr_classes=1)
        try:
            meter.measure(torch.LongTensor([0]), torch.LongTensor([[[0]]]))
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), meter.INVALID_BATCH_DIMENSION_MESSAGE.format(dims=3))

    def test_with_one_class_measures_returns_1x1_matrix_with_number_of_measured_tensors(self):
        meter = meters.ConfusionMatrix(nr_classes=1)
        self.assertMeasureEqual(meter, [(torch.LongTensor([0]), torch.LongTensor([0]))], 1*torch.ones(1,1))
        self.assertMeasureEqual(meter, [(torch.LongTensor([0, 0, 0]), torch.LongTensor([0, 0, 0]))], 3*torch.ones(1,1))
        self.assertMeasureEqual(meter, [(torch.LongTensor([0, 0]),    torch.LongTensor([0, 0])),
                                        (torch.LongTensor([0, 0, 0]), torch.LongTensor([0, 0, 0]))], 5*torch.ones(1,1))

    def test_can_not_measure_tensors_of_different_length(self):
        meter = meters.ConfusionMatrix(nr_classes=1)
        try:
            meter.measure(torch.LongTensor([0]), torch.LongTensor([0, 0]))
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), meter.INVALID_LENGTHS_MESSAGE)

    def test_can_not_measure_tensors_with_values_greater_than_nr_classes(self):
        meter = meters.ConfusionMatrix(nr_classes=2)
        try:
            meter.measure(torch.LongTensor([0, 1, 2]), torch.LongTensor([0, 0, 0]))
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), meter.INVALID_LABELS_MESSAGE)

        try:
            meter.measure(torch.LongTensor([0, 1, 1]), torch.LongTensor([1, 2, 0]))
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), meter.INVALID_LABELS_MESSAGE)

    def test_can_not_measure_tensors_with_values_less_than_zero(self):
        meter = meters.ConfusionMatrix(nr_classes=2)
        try:
            meter.measure(torch.LongTensor([0, 1, -1]), torch.LongTensor([0, 0, 0]))
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), meter.INVALID_LABELS_MESSAGE)

        try:
            meter.measure(torch.LongTensor([0, 1, 1]), torch.LongTensor([1, -1, 0]))
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), meter.INVALID_LABELS_MESSAGE)

    def test_meter_returns_zero_before_measurements(self):
        self.assertTensorsEqual(meters.ConfusionMatrix(nr_classes=2).value().matrix, torch.zeros(2, 2))

    def test_confusion_matrix_with_two_classes_count_one_each_class(self):
        meter = meters.ConfusionMatrix(nr_classes=2)
        self.assertMeasureEqual(meter, [(torch.LongTensor([0]), torch.LongTensor([0]))], torch.Tensor([[1, 0], [0, 0]]))
        self.assertMeasureEqual(meter, [(torch.LongTensor([1]), torch.LongTensor([1]))], torch.Tensor([[0, 0], [0, 1]]))
        self.assertMeasureEqual(meter, [(torch.LongTensor([0]), torch.LongTensor([1]))], torch.Tensor([[0, 1], [0, 0]]))
        self.assertMeasureEqual(meter, [(torch.LongTensor([1]), torch.LongTensor([0]))], torch.Tensor([[0, 0], [1, 0]]))

    def test_confusion_matrix_with_two_classes_count_one_each_class(self):
        meter = meters.ConfusionMatrix(nr_classes=2)
        self.assertMeasureEqual(meter, [(torch.LongTensor([0, 1]), torch.LongTensor([0, 0]))], torch.Tensor([[1, 0], [1, 0]]))
        self.assertMeasureEqual(meter, [(torch.LongTensor([1, 0]), torch.LongTensor([1, 0]))], torch.Tensor([[1, 0], [0, 1]]))
        self.assertMeasureEqual(meter, [(torch.LongTensor([0, 0]), torch.LongTensor([0, 0]))], torch.Tensor([[2, 0], [0, 0]]))
        self.assertMeasureEqual(meter, [(torch.LongTensor([1, 0]), torch.LongTensor([0, 1]))], torch.Tensor([[0, 1], [1, 0]]))

    def test_confusion_matrix_with_two_classes_count_one_each_class(self):
        meter = meters.ConfusionMatrix(nr_classes=3)
        self.assertMeasureEqual(meter, [(torch.LongTensor([0, 2, 1]), torch.LongTensor([0, 0, 2])),
                                        (torch.LongTensor([0]), torch.LongTensor([1])),
                                        (torch.LongTensor([2, 1, 2]), torch.LongTensor([1, 1, 2]))], torch.Tensor([[1, 1, 0], [0, 1, 1], [1, 1, 1]]))

    @requires_cuda
    def test_cannot_measure_with_cuda_float_tensors_on_left_param(self):
        meter = meters.ConfusionMatrix(nr_classes=1)
        a = torch.zeros(1).type(torch.FloatTensor).cuda()
        b = torch.zeros(1).type(torch.LongTensor).cuda()
        try:
            meter.measure(a, b)
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), meter.INVALID_INPUT_TYPE_MESSAGE.format(type_=a.type()))

    @requires_cuda
    def test_cannot_measure_with_cuda_float_tensors_on_right_param(self):
        meter = meters.ConfusionMatrix(nr_classes=1)
        a = torch.zeros(1).type(torch.FloatTensor).cuda()
        b = torch.zeros(1).type(torch.LongTensor).cuda()
        try:
            meter.measure(b, a)
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), meter.INVALID_INPUT_TYPE_MESSAGE.format(type_=a.type()))

    def test_confusion_matrix_with_auto_nr_of_class_infers_nr_classes(self):
        meter = meters.ConfusionMatrix(nr_classes='auto')
        self.assertMeasureEqual(meter, [(torch.LongTensor([0, 1]), torch.LongTensor([0, 1])),
                                        (torch.LongTensor([2, 1]), torch.LongTensor([1, 3]))], torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0]]))

    def test_cannot_construct_with_a_str_nr_classes_different_from_auto(self):
        try:
            meters.ConfusionMatrix(nr_classes='xyz')
            self.fail()
        except Exception as e:
            self.assertEqual(str(e), meters.ConfusionMatrix.INVALID_NR_OF_CLASSES_MESSAGE.format(nr_classes='xyz'))


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

class BalancedAccuracyTests(BaseMetricsTests):
    def test_final_value(self):
        meter = meters.BalancedAccuracy()

        self.assertMeasureAlmostEqual(meter, [(torch.Tensor([[1, 0, 0], # 0
                                                            [0, 1, 0],  # 1
                                                            [0, 1, 0],  # 1
                                                            [0, 0, 1]]), # 2
                                               torch.LongTensor([0,1,0,2]))], (1/2+1/1+1/1)/3)

class BinaryScores(BaseMetricsTests):
    def setUp(self):
        self.batch_1 = [(torch.LongTensor([1,1,1,0,0,1]),
                         torch.LongTensor([1,0,1,0,1,0]))]
        self.batch_2 = [(# Batch 1
                         torch.LongTensor([1,1,1]),
                         torch.LongTensor([1,0,1])),
                         ## Batch 2
                        (torch.LongTensor([0,0,1]),
                         torch.LongTensor([0,1,0]))]

    def test_1_batch_precision(self):
        meter = meters.Precision()
        self.assertMeasureAlmostEqual(meter, self.batch_1, 2/4)

    def test_1_batch_recall(self):
        meter = meters.Recall()
        self.assertMeasureAlmostEqual(meter, self.batch_1, 2/3)

    def test_2_batch_recall(self):
        meter = meters.Recall()
        meter.measure(*self.batch_2[0])
        self.assertAlmostEqual(meter.value(), 2/2)
        meter.measure(*self.batch_2[1])
        self.assertAlmostEqual(meter.value(), 2/3)
        meter.reset()
        meter.measure(*self.batch_2[1])
        self.assertAlmostEqual(meter.value(), 0)

    def test_1_batch_specifity(self):
        meter = meters.Specificity()
        self.assertMeasureAlmostEqual(meter, self.batch_1, 1/3)

    def test_1_batch_npv(self):
        meter = meters.NPV()
        self.assertMeasureAlmostEqual(meter, self.batch_1, 1/2)

    def test_1_batch_f1_score(self):
        meter = meters.F1Score()
        self.assertMeasureAlmostEqual(meter, self.batch_1, 2 * (2/4 * 2/3) / (2/4 + 2/3))

class SpeedMetersTests(BaseMetricsTests):
    def test_default_mode(self):
        meters = parse_meters({'bps': 'batches/sec',
                               'spb': 'sec/batch',
                               'ips': 'it/sec',
                               'spi': 'sec/it',

                               'bpm': 'batches/min',
                               'mpb': 'min/batch',
                               'ipm': 'it/min',
                               'mpi': 'min/it',
                               })
        for meter in ['bps', 'bpm', 'ips', 'ipm']:
            self.assertEqual(meters[meter].DEFAULT_MODE, 'max')
        for meter in ['spb', 'mpb', 'spi', 'mpi']:
            self.assertEqual(meters[meter].DEFAULT_MODE, 'min')

    def test_batch_speed_is_less_than_sleep_time(self):
        meters = parse_meters({'bps': 'batches/sec',
                               'spb': 'sec/batch',
                               'ips': 'it/sec',
                               'spi': 'sec/it',

                               'bpm': 'batches/min',
                               'mpb': 'min/batch',
                               'ipm': 'it/min',
                               'mpi': 'min/it',
                               })

        for meter in meters.values():
            meter.measure(torch.zeros(3, 5))
        sleep(0.5)
        for meter in meters.values():
            meter.measure(torch.zeros(3, 5))
        sleep(1)
        for meter in meters.values():
            meter.measure(torch.zeros(2, 5))
        sleep(1.5)
        total_time = (0.5+1+1.5)
        total_batches = 3
        total_it = 7
        self.assertGreaterEqual(meters['bps'].value(), total_batches/total_time)
        self.assertGreaterEqual(meters['bpm'].value(), total_batches/(total_time/60))
        self.assertGreaterEqual(meters['ips'].value(), total_it/total_time)
        self.assertGreaterEqual(meters['ipm'].value(), total_it/(total_time*60))

        self.assertLessEqual(meters['spb'].value(), total_time/total_batches)
        self.assertLessEqual(meters['mpb'].value(), (total_time/60)/total_batches)
        self.assertLessEqual(meters['spi'].value(), total_time/total_it)
        self.assertLessEqual(meters['mpi'].value(), (total_time/60)/total_it)
