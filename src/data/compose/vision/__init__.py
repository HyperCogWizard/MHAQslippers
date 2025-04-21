from .mnist import MNISTDataModule as MNIST
from .cifar10 import CIFAR10DataModule as CIFAR10
from .cifar100 import CIFAR100DataModule as CIFAR100
from .imagenet import ImageNetDataModule as IMAGENET
from .cifar10_dali import CIFAR10DALIDataModule as CIFAR10_DALI
from .cifar10_noise import CIFAR10NOISEDataModule as CIFAR10_NOISE
from .cifar100_dali import CIFAR10DALIDataModule as CIFAR100_DALI
from .imagenet_dali import ImageNetDALIDataModule as IMAGENET_DALI
from .cifar100_noise import CIFAR100NOISEDataModule as CIFAR100_NOISE

__all__ = [
    "MNIST",
    "CIFAR10",
    "CIFAR10_DALI",
    "CIFAR10_NOISE",
    "CIFAR100",
    "CIFAR100_DALI",
    "CIFAR100_NOISE",
    "IMAGENET",
    "IMAGENET_DALI",
]
