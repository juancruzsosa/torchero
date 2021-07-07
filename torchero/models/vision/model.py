from torchvision import transforms
from torchero.models import (Model,
                             BinaryClassificationModel,
                             ClassificationModel,
                             RegressionModel)

class ImageModel(Model):
    """ Model class that wrap nn.Module models to add
    training, prediction, saving & loading capabilities
    for Computer Vision tasks
    """
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
