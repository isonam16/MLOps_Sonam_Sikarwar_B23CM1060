import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np


class CIFAR10CustomDataset(Dataset):
    """Custom CIFAR-10 Dataset wrapper with additional functionality"""
    
    def __init__(self, root='./data', train=True, transform=None, download=True):
        self.cifar10 = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=None
        )
        self.transform = transform
        
    def __len__(self):
        return len(self.cifar10)
    
    def __getitem__(self, idx):
        image, label = self.cifar10[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_cifar10_dataloaders(batch_size=128, num_workers=2):
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    # Define transformations for test data (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    # Create custom datasets
    train_dataset = CIFAR10CustomDataset(
        root='./data',
        train=True,
        transform=train_transform,
        download=True
    )
    
    test_dataset = CIFAR10CustomDataset(
        root='./data',
        train=False,
        transform=test_transform,
        download=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Testing batches: {len(test_loader)}")
    
    return train_loader, test_loader


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
