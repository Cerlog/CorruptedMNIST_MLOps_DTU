from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

if TYPE_CHECKING:
    import torchvision.transforms.v2 as transforms


class CorruptMNISTDataset(Dataset):
    """Corrupt MNIST dataset for PyTorch.

    Args:
        data_folder: Path to the data folder.
        train: Whether to load training or test data.
        raw_folder: Optional path to raw data folder for preprocessing.
        processed_folder: Optional path to processed data folder.
        img_transform: Image transformation to apply.
        target_transform: Target transformation to apply.
    """

    name: str = "CorruptMNIST"

    def __init__(
        self,
        data_folder: str = "data",
        train: bool = True,
        raw_folder: Optional[str] = None,
        processed_folder: Optional[str] = None,
        img_transform: Optional[transforms.Transform] = None,
        target_transform: Optional[transforms.Transform] = None,
    ) -> None:
        super().__init__()
        self.data_folder = data_folder
        self.train = train
        self.raw_folder = raw_folder or os.path.join(data_folder, "raw")
        self.processed_folder = processed_folder or os.path.join(data_folder, "processed")
        self.img_transform = img_transform
        self.target_transform = target_transform
        
        # Create processed folder if it doesn't exist
        if not os.path.exists(self.processed_folder):
            os.makedirs(self.processed_folder)
            
        # Preprocess data if necessary
        if raw_folder and not self._check_processed_files():
            self.preprocess_data()
            
        self.load_data()

    def _check_processed_files(self) -> bool:
        """Check if processed files exist."""
        files = ["train_images.pt", "train_target.pt", "test_images.pt", "test_target.pt"]
        return all(os.path.exists(os.path.join(self.processed_folder, f)) for f in files)

    @staticmethod
    def normalize(images: Tensor) -> Tensor:
        """Normalize images by subtracting mean and dividing by standard deviation."""
        return (images - images.mean()) / images.std()

    def preprocess_data(self) -> None:
        """Load raw data, preprocess (normalize) it, and save to processed directory."""
        # Load and concatenate training data
        train_images, train_target = [], []
        for i in range(6):  # Load six parts of training data
            train_images.append(torch.load(f"{self.raw_folder}/train_images_{i}.pt"))
            train_target.append(torch.load(f"{self.raw_folder}/train_target_{i}.pt"))
        train_images = torch.cat(train_images)
        train_target = torch.cat(train_target)

        # Load test data
        test_images = torch.load(f"{self.raw_folder}/test_images.pt")
        test_target = torch.load(f"{self.raw_folder}/test_target.pt")

        # Reshape and convert types
        train_images = train_images.unsqueeze(1).float()
        test_images = test_images.unsqueeze(1).float()
        train_target = train_target.long()
        test_target = test_target.long()

        # Normalize images
        train_images = self.normalize(train_images)
        test_images = self.normalize(test_images)

        # Save processed data
        torch.save(train_images, f"{self.processed_folder}/train_images.pt")
        torch.save(train_target, f"{self.processed_folder}/train_target.pt")
        torch.save(test_images, f"{self.processed_folder}/test_images.pt")
        torch.save(test_target, f"{self.processed_folder}/test_target.pt")

    def load_data(self) -> None:
        """Load preprocessed images and targets from disk."""
        if self.train:
            self.images = torch.load(f"{self.processed_folder}/train_images.pt")
            self.target = torch.load(f"{self.processed_folder}/train_target.pt")
        else:
            self.images = torch.load(f"{self.processed_folder}/test_images.pt")
            self.target = torch.load(f"{self.processed_folder}/test_target.pt")

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Return image and target tensor for given index."""
        img, target = self.images[idx], self.target[idx]
        
        if self.img_transform:
            img = self.img_transform(img)
        if self.target_transform:
            target = self.target_transform(target)
            
        return img, target

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return self.images.shape[0]


# Helper function to get both train and test datasets
def get_corrupt_mnist(
    data_folder: str = "data",
    raw_folder: Optional[str] = None,
    processed_folder: Optional[str] = None,
    img_transform: Optional[transforms.Transform] = None,
    target_transform: Optional[transforms.Transform] = None,
) -> Tuple[Dataset, Dataset]:
    """Return the train and test datasets for the Corrupt MNIST dataset."""
    train_set = CorruptMNISTDataset(
        data_folder=data_folder,
        train=True,
        raw_folder=raw_folder,
        processed_folder=processed_folder,
        img_transform=img_transform,
        target_transform=target_transform,
    )
    test_set = CorruptMNISTDataset(
        data_folder=data_folder,
        train=False,
        raw_folder=raw_folder,
        processed_folder=processed_folder,
        img_transform=img_transform,
        target_transform=target_transform,
    )
    return train_set, test_set

def show_image_and_target(images: torch.Tensor, target: torch.Tensor, show: bool = True) -> None:
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    if show:
        plt.show()


if __name__ == "__main__":
    # Example usage
    train_dataset, test_dataset = get_corrupt_mnist(
        raw_folder="data/raw",
        processed_folder="data/processed"
    )
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")