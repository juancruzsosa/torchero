from torchvision import transforms
from torchero.models import (Model,
                             BinaryClassificationModel,
                             ClassificationModel,
                             RegressionModel)

class ImageModel(Model):
    def __init__(self, model, transform=None):
        super(ImageModel, self).__init__(model)
        if transform is None:
            transform = transforms.ToTensor()
        self.transform = transform

    def input_to_tensor(self, image):
        return self.transform(image)

class BinaryImageClassificationModel(ImageModel, BinaryClassificationModel):
    def __init__(self, model, transform, use_logits=True, threshold=0.5):
        super(BinaryImageClassificationModel, self).__init__(model=model,
                                                             transform=transform)
        super(ImageModel, self).__init__(model=model,
                                         use_logits=use_logits,
                                         threshold=threshold)

class ImageClassificationModel(ImageModel, ClassificationModel):
    def __init__(self, model, transform, use_softmax=True, threshold=0.5):
        super(ImageClassificationModel, self).__init__(model=model,
                                                       transform=transform)
        super(ImageModel, self).__init__(model=model,
                                         use_softmax=use_softmax)

class ImageRegressionModel(ImageModel, RegressionModel):
    def __init__(self, model, transform, use_logits=False, threshold=0.5):
        super(ImageRegressionModel, self).__init__(model=model,
                                         transform=transform)
