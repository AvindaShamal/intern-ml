from torch.utils.data import Dataset
from typing import Callable
from datasets import DatasetDict


class HuggingFaceDataset(Dataset):
    """A custom dataset class for loading images from a directory structure where each
    class has its own subdirectory.
    """

    def __init__(
        self,
        hf_dataset: DatasetDict,
        split: str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        """Initializes the CustomImageDataset.

        Args:
            hf_dataset : Huggingface dataset object
            split : The split to use ('train', 'test', 'validation')
            transform : A function to apply to the images (default: None).
            target_transform : A function to apply to the labels (default: None).
        Returns: None
        """
        self.dataset = hf_dataset
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.dataset[self.split])

    def __getitem__(self, idx: int) -> tuple:
        """Retrieves the image and label at the specified index.

        Args:
            idx : Index of the sample to retrieve.
        Returns:
            A tuple containing the image tensor and its corresponding label.
        """
        data = self.dataset[self.split][idx]
        image, label = data["img"], data["label"]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
