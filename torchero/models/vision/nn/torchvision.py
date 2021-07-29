from torch import nn
from torchvision import models

arch_aliases = {
    'alexnet': models.alexnet,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'resnext50_32x4d': models.resnext50_32x4d,
    'resnext101_32x8d': models.resnext101_32x8d,
    'wide_resnet50_2': models.wide_resnet50_2,
    'wide_resnet101_2': models.wide_resnet101_2,
    'vgg11': models.vgg11,
    'vgg11_bn': models.vgg11_bn,
    'vgg13': models.vgg13,
    'vgg13_bn': models.vgg13_bn,
    'vgg16': models.vgg16,
    'vgg16_bn': models.vgg16_bn,
    'vgg19_bn': models.vgg19_bn,
    'vgg19': models.vgg19,
    'squeezenet1_0': models.squeezenet1_0,
    'squeezenet1_1': models.squeezenet1_1,
    'densenet121': models.densenet121,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    'densenet161': models.densenet161,
    'mobilenet_v2': models.mobilenet_v2,
    'mnasnet0_5': models.mnasnet0_5,
    'mnasnet0_75': models.mnasnet0_75,
    'mnasnet1_0': models.mnasnet1_0,
    'mnasnet1_3': models.mnasnet1_3,
    'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5,
    'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0,
    'shufflenet_v2_x1_5': models.shufflenet_v2_x1_5,
    'shufflenet_v2_x2_0': models.shufflenet_v2_x2_0,
}

class TorchvisionModel(nn.Module):
    @classmethod
    def from_config(cls, config):
        return cls(config['arch'],
                   config['num_outputs'],
                   pretrained=True)

    def patch_classifier_submodule(self):
        model = self.model
        if isinstance(model, (models.AlexNet, models.VGG, models.MobileNetV2, models.MNASNet)):
            assert(isinstance(model.classifier[-1], nn.Linear))
            model.classifier[-1] = nn.Linear(model.classifier[-1]
                                                  .in_features,
                                             num_outputs)
        elif isinstance(model, (models.ResNet, models.ShuffleNetV2)):
            assert(isinstance(model.fc, nn.Linear))
            model.fc = nn.Linear(model.fc.in_features,
                                 num_outputs)
        elif isinstance(model, models.SqueezeNet):
            assert(isinstance(model.classifier[1], nn.Conv2d))
            model.classifier[1] = nn.Conv2d(512,
                                            num_outputs,
                                            kernel_size=1)
        elif isinstance(model, models.DenseNet):
            assert(isinstance(model.classifier, nn.Linear))
            model.classifier = nn.Linear(model.classifier
                                              .in_features,
                                         num_outputs)
        else:
            raise NotImplemented("Architecture {} not supported.".format(arch.__class__.__name__))

    def __init__(self, arch, num_outputs=None, pretrained=False):
        """ Constructor

        Arguments:
            arch (str): Architecture name. See arch_aliases
            num_outputs (int, optional): Set to None to use the same number of
                outputs of the torchvision model (1000). Set to a number to change
                the number of outputs by changing the classifier layer
            pretrained (bool): Set to true to download the pretrained model for ILSVRC competition (ImageNet)
        """
        if arch not in arch_aliases:
            raise KeyError("wrong architecture. select from {}".format(',\n\t'.join(arch_aliases.keys())))
        self.arch = arch
        self.num_outputs = num_outputs or 1000
        self.model = arch_aliases[arch](pretrained=pretrained)
        if num_outputs is not None:
            self.patch_classifier_submodule()

    @property
    def config(self):
        return {'arch': self.arch}

    def forward(self, x):
        return self.model(x)
