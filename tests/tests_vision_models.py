from .common import *
from torchero.models.vision.nn import TorchvisionModel, arch_aliases

arch_dims = {
    'alexnet': 4096,
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
    'resnext50_32x4d': 2048,
    'resnext101_32x8d': 2048,
    'wide_resnet50_2': 2048,
    'wide_resnet101_2': 2048,
    'vgg11': 4096,
    'vgg11_bn': 4096,
    'vgg13': 4096,
    'vgg13_bn': 4096,
    'vgg16': 4096,
    'vgg16_bn': 4096,
    'vgg19_bn': 4096,
    'vgg19': 4096,
    'squeezenet1_0': 512,
    'squeezenet1_1': 512,
    'densenet121': 1024,
    'densenet169': 1664,
    'densenet201': 1920,
    'densenet161': 2208,
    'mobilenet_v2': 1280,
    'mnasnet0_5': 1280,
    'mnasnet0_75': 1280,
    'mnasnet1_0': 1280,
    'mnasnet1_3': 1280,
    'shufflenet_v2_x0_5': 1024,
    'shufflenet_v2_x1_0': 1024,
    'shufflenet_v2_x1_5': 1024,
    'shufflenet_v2_x2_0': 2048,
}

class ComputerVisionNNTest(unittest.TestCase):
    def setUp(self):
        self.archs = arch_aliases.keys()

    def assertTensorsEqual(self, a, b):
        return self.assertEqual(a.tolist(), b.tolist())

    def test_predict_shape(self):
        for arch in self.archs:
            with self.subTest(arch=arch):
                net = TorchvisionModel(arch, num_outputs=11, pretrained=False)
                net.train(False)
                x = torch.rand(2, 3, 224, 224)
                outputs = net(x)
                self.assertEqual(outputs.shape, torch.Size([2, 11]))

    def test_embedding_shape(self):
        for arch in self.archs:
            with self.subTest(arch=arch):
                net = TorchvisionModel(arch, num_outputs=11, pretrained=False)
                net.train(False)
                x = torch.rand(2, 3, 224, 224)
                outputs = net(x)
                embed, outputs2 = net.embed(x)
                self.assertTensorsEqual(outputs2, outputs)
                self.assertEqual(embed.shape, torch.Size([2, arch_dims[arch]]))

    def test_config(self):
        net = TorchvisionModel('resnet18')
        config1 = net.config
        net = TorchvisionModel.from_config(config1)
        config2 = net.config
        self.assertEqual(config1, config2)
