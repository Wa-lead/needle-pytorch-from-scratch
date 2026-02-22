import os
import pickle
import numpy as np
from typing import Optional, List
from ..data_basic import Dataset


class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset loader.

    Expects the dataset in the original Python pickle format
    (data_batch_1, ..., data_batch_5, test_batch).
    """

    def __init__(self, base_folder: str, train: bool = True, transforms: Optional[List] = None):
        super().__init__(transforms)
        if train:
            filenames = [os.path.join(base_folder, f"data_batch_{i}") for i in range(1, 6)]
        else:
            filenames = [os.path.join(base_folder, "test_batch")]

        images, labels = [], []
        for filepath in filenames:
            with open(filepath, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            images.append(batch[b"data"])
            labels.append(np.array(batch[b"labels"]))

        self.images = np.concatenate(images).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        self.labels = np.concatenate(labels)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        if img.ndim == 3 and self.transforms is not None:
            img = img.transpose(1, 2, 0)  # CHW -> HWC for transforms
            img = self.apply_transforms(img)
            img = img.transpose(2, 0, 1)  # HWC -> CHW
        return img, label

    def __len__(self):
        return len(self.images)
