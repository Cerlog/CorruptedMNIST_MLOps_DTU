import torch
import typer


def normalize(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batch of images by subtracting the mean and dividing by the standard deviation.

    Args:
        images (torch.Tensor): Batch of images.

    Returns:
        torch.Tensor: Normalized images.
    """
    # Subtract mean and divide by std to normalize images
    return (images - images.mean()) / images.std()


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """
    Load raw data, preprocess (normalize) it, and save it to the processed directory.

    Args:
        raw_dir (str): Directory containing the raw data.
        processed_dir (str): Directory where the processed data should be saved.
    """
    # Lists to store training images and targets
    train_images, train_target = [], []

    # Load six parts of training data
    for i in range(6):
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{raw_dir}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # Load test images and targets
    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt")

    # Reshape and type conversion
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    # Normalize both train and test images
    train_images = normalize(train_images)
    test_images = normalize(test_images)

    # Save processed data
    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Return the train and test datasets for the Corrupt MNIST dataset.

    Returns:
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        Tuple of training and testing datasets.
    """
    # Load preprocessed data
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    # Create TensorDatasets for train and test
    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess_data)
