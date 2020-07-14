from torchero.supervised_trainer import SupervisedTrainer, SupervisedValidator


class AutoencoderValidator(SupervisedValidator):
    def validate_batch(self, x):
        super(AutoencoderValidator, self).validate_batch(x, x)


class AutoencoderTrainer(SupervisedTrainer):
    """ Autoencoder trainer
    """
    def create_validator(self, metrics):
        return AutoencoderValidator(self.model, metrics)

    def update_batch(self, x):
        super(AutoencoderTrainer, self).update_batch(x, x)
