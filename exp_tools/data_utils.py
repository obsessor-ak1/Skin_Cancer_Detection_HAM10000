from math import ceil
from pathlib import Path
import shutil

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class HAM10000Dataset(Dataset):
    """This class loads the HAM10000 dataset from the specified directory."""

    def __init__(self, dataset_dir="./data", transform=None, target_transform=None,
                 download=False, split=None):
        # Downloading the dataset if requested or not found
        if download:
            download_ham10000(path=dataset_dir)
        self._data_path = Path(dataset_dir)
        self.metadata = pd.read_csv(self._data_path / "HAM10000_metadata.csv")
        # Getting all the possible labels
        labels = self.metadata.dx.unique()
        # Performing a train test split if requested.
        self.metadata = self.metadata.sample(frac=1, random_state=1).reset_index(drop=True)
        if split:
            mt_train, mt_test = train_test_split(
                self.metadata,
                test_size=0.2,
                stratify=self.metadata.dx,
                random_state=42
            )
            self.metadata = mt_train if split == "train" else mt_test
        self.metadata.reset_index(drop=True, inplace=True)
        self._transform = transform
        self._target_transform = target_transform
        self.label_map = dict(zip(labels, range(len(labels))))
        self._part1_folder = self._data_path / "HAM10000_images_part_1"
        self._part2_folder = self._data_path / "HAM10000_images_part_2"
        self._image_paths = self._load_image_paths()

    def _load_image_paths(self):
        """Pre-loads the paths for all the images in the dataset."""
        path_list = []
        for image_id in self.metadata.image_id:
            name = f"{image_id}.jpg"
            path1 = self._part1_folder / name
            path2 = self._part2_folder / name
            valid_path = path1 if path1.exists() else path2
            path_list.append(valid_path)
        return path_list

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        label = self.metadata.dx[idx]
        img_id, label = self.metadata.loc[idx, ["image_id", "dx"]]
        image_path = self._image_paths[idx]
        image = read_image(str(image_path))
        label = self.label_map[label]
        if self._transform:
            image = self._transform(image)
        if self._target_transform:
            label = self._target_transform(label)
        return image, label

    @property
    def sample_dist(self):
        props = self.metadata.dx.value_counts(normalize=True)
        props = props.loc[list(self.label_map.keys())]
        return torch.tensor(props.to_numpy())


class Explorer:
    """A class helps to explore and visualize the dataset."""

    def __init__(self, dataset):
        self.dataset = dataset

    def show_distribution(self, criteria="dx"):
        data = self.dataset.metadata[criteria]
        counts = data.value_counts()
        plt.bar(counts.index, counts)

    def show_images(self, criteria="dx", value="", max_count=10):
        data = self.dataset.metadata[criteria]
        if value:
            data = data[data == value]
        data = data.sample(n=max_count, replace=True)
        images = self._load_images(data.index)
        n_rows = ceil(len(data) / 4)
        plt.figure(figsize=(12, 3 * n_rows))
        for i, (label, image) in enumerate(zip(data, images)):
            plt.subplot(n_rows, 4, i + 1)
            image = image.permute(1, 2, 0)
            plt.imshow(image)
            plt.title(label)

    def _load_images(self, ids):
        images = []
        for idx in ids:
            image, _ = self.dataset[idx]
            images.append(image)
        return images


def download_ham10000(path="./data", force_download=False):
    """Downloads the HAM10000 dataset from kaggle."""
    default_path = kagglehub.dataset_download(
        "kmader/skin-cancer-mnist-ham10000",
        force_download=force_download
    )
    print("Downloaded to:", default_path)
    shutil.move(default_path, path)
    print("Moved to:", path)
