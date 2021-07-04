from abc import ABCMeta, abstractmethod

import torch

from torchero.meters.base import BaseMeter


class ConfusionMatrixController(object, metaclass=ABCMeta):
    def __init__(self, normalize=False):
        self.normalize = normalize
        self.reset()

    @property
    @abstractmethod
    def matrix(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def increment(self, a, b):
        for i, j in zip(a, b):
            self.matrix[i][j] += 1

    @property
    def num_classes(self):
        return self.matrix.shape[0]

    def plot(self, ax=None, fig=None, classes=None, xlabel="Predicted label", ylabel="True label", title="Confusion Matrix", cmap="Blues", colorbar=False):
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib is required in order to plot confusion matrix"
            )
        if (classes is not None) and (len(classes) != self.num_classes):
            raise ValueError("number of classes is: {} but {} were passed!".format(self.num_classes, len(classes)))

        if ax is None:
            ax = plt.gca()
        if fig is None:
            fig = plt.gcf()
        matrix = self.matrix
        normalized_matrix = matrix / matrix.sum(dim=0)
        cmap = plt.get_cmap(cmap)
        im=ax.imshow(normalized_matrix, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(self.num_classes))
        ax.set_yticks(range(self.num_classes))

        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j].item()
                normalized_value = normalized_matrix[i, j].item()
                if self.normalize:
                    value = '{:.2f}'.format(value)
                else:
                    value = '{}'.format(int(value))
                    if i == j:
                        value += " " + "({:.0f}%)".format(normalized_value * 100)
                r, g, b, _ = cmap(normalized_value)
                text_color = 'white' if r * g * b < 0.5 else 'black'
                text = ax.text(j, i, value,
                               ha="center", va="center", color=text_color)

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        if classes is not None:
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)

        if colorbar:
            fig.colorbar(im, ax=ax)


class FixedConfusionMatrixController(ConfusionMatrixController):
    def __init__(self, nr_classes, normalize=False):
        if not isinstance(nr_classes, int) or nr_classes == 0:
            raise Exception(ConfusionMatrix.INVALID_NR_OF_CLASSES_MESSAGE
                                           .format(nr_classes=nr_classes))
        self._nr_classes = nr_classes
        super(FixedConfusionMatrixController, self).__init__(normalize=normalize)

    @property
    def matrix(self):
        if self.normalize:
            return self._matrix / self._matrix.sum(dim=0)
        else:
            return self._matrix

    def reset(self):
        self._matrix = torch.zeros(self._nr_classes, self._nr_classes)

    def check_inputs(self, xs):
        if not ((0 <= xs) & (xs < self._matrix.shape[0])).all():
            raise Exception(ConfusionMatrix.INVALID_LABELS_MESSAGE)

    def increment(self, a, b):
        self.check_inputs(torch.cat([a, b]))
        super(FixedConfusionMatrixController, self).increment(a, b)


class ResizableConfusionMatrixController(ConfusionMatrixController):
    @property
    def matrix(self):
        if self.normalize:
            return self._matrix / self._matrix.sum(dim=0)
        else:
            return self._matrix

    def reset(self):
        self._matrix = torch.zeros(1, 1)

    def expand(self, n):
        total_rows = n + self._matrix.shape[0]
        total_cols = n + self._matrix.shape[1]

        old_matrix, self._matrix = self._matrix, torch.zeros(total_rows,
                                                             total_cols)
        self._matrix[:old_matrix.shape[0], :old_matrix.shape[1]] = old_matrix

    def increment(self, a, b):
        max_class_nr = max(torch.max(a), torch.max(b))

        if max_class_nr >= self._matrix.shape[0]:
            n = max_class_nr - self._matrix.shape[0] + 1
            self.expand(n)

        super(ResizableConfusionMatrixController, self).increment(a, b)

class ConfusionMatrix(BaseMeter):
    INVALID_NR_OF_CLASSES_MESSAGE = (
        'Expected number of classes to be greater than one. Got {nr_classes}'
    )
    INVALID_INPUT_TYPE_MESSAGE = (
        'Expected input tensors of type LongTensor. Got {type_}'
    )
    INVALID_BATCH_DIMENSION_MESSAGE = (
        'Expected input tensors of 1-dimention. Got {dims}'
    )
    INVALID_LENGTHS_MESSAGE = (
        'Expected input and targets of same lengths'
    )
    INVALID_LABELS_MESSAGE = (
        'Expected labels between 0 and number of classes'
    )

    def __init__(self, nr_classes='auto', normalize=False):
        """ Constructor

        Arguments:
            nr_classes (int or str): If 'auto' is passed the confusion matrix will readjust
                to the observed ranges. If a number is passed this will reserve the confusion matrix
                for that size. Default 'auto'
            normalize (bool): IF passed the confusion matrix will hold percentages
        """
        if isinstance(nr_classes, str) and nr_classes == 'auto':
            self.matrix_controller = ResizableConfusionMatrixController(normalize=normalize)
        elif isinstance(nr_classes, int) and nr_classes > 0:
            self.matrix_controller = FixedConfusionMatrixController(nr_classes, normalize=normalize)
        else:
            raise ValueError(self.INVALID_NR_OF_CLASSES_MESSAGE
                                 .format(nr_classes=nr_classes))
        self.reset()

    def reset(self):
        self.matrix_controller.reset()

    def check_tensor(self, a):
        if (isinstance(a, torch.FloatTensor) or
                isinstance(a, torch.cuda.FloatTensor)):
            raise Exception(self.INVALID_INPUT_TYPE_MESSAGE
                                .format(type_=a.type()))

        if a.dim() > 1:
            raise Exception(self.INVALID_BATCH_DIMENSION_MESSAGE
                                .format(dims=a.dim()))

    def measure(self, a, b):
        if a.dim() == 2:
            a = a.topk(k=1, dim=1)[1].squeeze(1)
        if b.dim() == 2:
            b = b.topk(k=1, dim=1)[1].squeeze(1)

        self.check_tensor(a)
        self.check_tensor(b)

        if len(a) != len(b):
            raise Exception(self.INVALID_LENGTHS_MESSAGE)

        self.matrix_controller.increment(a, b)

    def value(self):
        return self.matrix_controller
