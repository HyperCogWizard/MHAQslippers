from .mnist import MNISTDataModule as MNIST
from .cifar10 import CIFAR10DataModule as CIFAR10
from .cifar100 import CIFAR100DataModule as CIFAR100
from .imagenet import ImageNetDataModule as IMAGENET
from .cifar10_dali import CIFAR10DALIDataModule as CIFAR10_DALI
from .cifar100_dali import CIFAR10DALIDataModule as CIFAR100_DALI
from .imagenet_dali import ImageNetDALIDataModule as IMAGENET_DALI
from .sr import Div2K, Set5, Set14, B100, Urban100

__all__ = [
    "MNIST",
    "CIFAR10",
    "CIFAR10_DALI",
    "CIFAR100",
    "CIFAR100_DALI",
    "IMAGENET",
    "IMAGENET_DALI",
]
