import torch
from lightning import pytorch as pl

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.transforms import AutoAugmentPolicy

from torch.utils.data import random_split, DataLoader, Dataset

class UniformNoiseDataset(Dataset):
    def __init__(self, size, num_classes=10):
        self.size = size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate uniform noise in the range [0, 1]
        noise = torch.rand(3, 32, 32)  # Shape matches CIFAR-10 images
        label = torch.randint(0, self.num_classes, (1,)).item()
        return noise, label

class CIFAR10NOISEDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = "./data",
                 batch_size=1000,
                 num_workers=5) -> None:

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform_train = transforms.Compose(
            [
                # transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                self._normalize(),
            ]
        )

        self.transform_test = transforms.Compose(
            [transforms.ToTensor(), self._normalize()]
        )

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        cifar_data = CIFAR10(
            self.data_dir, train=True, transform=self.transform_train
        )

        self.cifar_train = UniformNoiseDataset(size=45000)
        
        _, self.cifar_val = random_split(
            cifar_data, [45000, 5000], generator=torch.Generator().manual_seed(42)
        )

        self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform_test
            )

        if stage == "predict":
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform_test
            )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=5,
            shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    @staticmethod
    def _normalize():
        return transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]
        )
