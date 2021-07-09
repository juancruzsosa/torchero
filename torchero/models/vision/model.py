import torch
from torch import nn

from torchvision import transforms
from torchero.models import (Model,
                             BinaryClassificationModel,
                             ClassificationModel,
                             RegressionModel)
from torchero.models.vision.nn import pretrained, torchvision

__all__ = ['arch_aliases',
           'ImageModel',
           'BinaryImageClassificationModel',
           'ImageClassificationModel',
           'ImageRegressionModel']

arch_aliases = {
    'alexnet': pretrained.alexnet,
    'resnet18': pretrained.resnet18,
    'resnet34': pretrained.resnet34,
    'resnet50': pretrained.resnet50,
    'resnet101': pretrained.resnet101,
    'resnet152': pretrained.resnet152,
    'resnext50_32x4d': pretrained.resnext50_32x4d,
    'resnext101_32x8d': pretrained.resnext101_32x8d,
    'wide_resnet50_2': pretrained.wide_resnet50_2,
    'wide_resnet101_2': pretrained.wide_resnet101_2,
    'vgg11': pretrained.vgg11,
    'vgg11_bn': pretrained.vgg11_bn,
    'vgg13': pretrained.vgg13,
    'vgg13_bn': pretrained.vgg13_bn,
    'vgg16': pretrained.vgg16,
    'vgg16_bn': pretrained.vgg16_bn,
    'vgg19_bn': pretrained.vgg19_bn,
    'vgg19': pretrained.vgg19,
    'squeezenet1_0': pretrained.squeezenet1_0,
    'squeezenet1_1': pretrained.squeezenet1_1,
    'densenet121': pretrained.densenet121,
    'densenet169': pretrained.densenet169,
    'densenet201': pretrained.densenet201,
    'densenet161': pretrained.densenet161,
    'mobilenet_v2': pretrained.mobilenet_v2,
    'mnasnet0_5': pretrained.mnasnet0_5,
    'mnasnet0_75': pretrained.mnasnet0_75,
    'mnasnet1_0': pretrained.mnasnet1_0,
    'mnasnet1_3': pretrained.mnasnet1_3,
    'shufflenet_v2_x0_5': pretrained.shufflenet_v2_x0_5,
    'shufflenet_v2_x1_0': pretrained.shufflenet_v2_x1_0,
    'shufflenet_v2_x1_5': pretrained.shufflenet_v2_x1_5,
    'shufflenet_v2_x2_0': pretrained.shufflenet_v2_x2_0,
}

class ImageModel(Model):
    """ Model class that wrap nn.Module models to add
    training, prediction, saving & loading capabilities
    for Computer Vision tasks
    """

    @classmethod
    def from_pretrained(cls, arch, num_outputs=None, transform=transforms.ToTensor()):
        if arch not in arch_aliases:
            raise Exception("wrong architecture. select from {}".format(',\n\t'.join(arch_aliases.keys())))

        model = arch_aliases[arch](pretrained=True)
        if num_outputs is not None:
            if isinstance(model, (torchvision.AlexNet, torchvision.VGG, torchvision.MobileNetV2, torchvision.MNASNet)):
                assert(isinstance(model.classifier[-1], nn.Linear))
                model.classifier[-1] = nn.Linear(model.classifier[-1]
                                                      .in_features,
                                                 num_outputs)
            elif isinstance(model, (torchvision.ResNet, torchvision.ShuffleNetV2)):
                assert(isinstance(model.fc, nn.Linear))
                model.fc = nn.Linear(model.fc.in_features,
                                     num_outputs)
            elif isinstance(model, torchvision.SqueezeNet):
                assert(isinstance(model.classifier[1], nn.Conv2d))
                model.classifier[1] = nn.Conv2d(512,
                                                num_outputs,
                                                kernel_size=1)
            elif isinstance(model, torchvision.DenseNet):
                assert(isinstance(model.classifier, nn.Linear))
                model.classifier = nn.Linear(model.classifier
                                                  .in_features,
                                             num_outputs)
            else:
                raise NotImplemented("Architecture not supported")
        return cls(model, transform=transform)

    def __init__(self, model, transform=None):
        super(ImageModel, self).__init__(model)
        if transform is None:
            transform = transforms.ToTensor()
        self.transform = transform

    def _save_to_zip(self, zip_fp):
        super(ImageModel, self)._save_to_zip(zip_fp)
        with zip_fp.open('transform.pkl', 'w') as fp:
            pickle.dump(self.transform, fp)

    def input_to_tensor(self, image):
        return self.transform(image)

class BinaryImageClassificationModel(ImageModel, BinaryClassificationModel):
    """ Model class for Image Binary Classification (single or multilabel) tasks.
    E.g: distinguish real vs fake images
    """
    def __init__(self, model, transform, use_logits=True, threshold=0.5):
        super(BinaryImageClassificationModel, self).__init__(model=model,
                                                             transform=transform)
        super(ImageModel, self).__init__(model=model,
                                         use_logits=use_logits,
                                         threshold=threshold)

class ImageClassificationModel(ImageModel, ClassificationModel):
    """ Model Class for Image Classification (for categorical targets) tasks.
    E.g: Predict ImageNet classes of a given image
    """
    def __init__(self, model, transform, use_softmax=True, threshold=0.5):
        super(ImageClassificationModel, self).__init__(model=model,
                                                       transform=transform)
        super(ImageModel, self).__init__(model=model,
                                         use_softmax=use_softmax)

class ImageRegressionModel(ImageModel, RegressionModel):
    """ Model Class for Image Regression tasks.
    E.g: Face Landmarks recognition
    """
    def __init__(self, model, transform, use_logits=False, threshold=0.5):
        super(ImageRegressionModel, self).__init__(model=model,
                                         transform=transform)
