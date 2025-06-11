from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class HAM10000Dataset(Dataset):
    """This class loads the HAM10000 dataset from the specified directory."""

    def __init__(self, dataset_dir="./data", start=0, count=8000, transform=None):
        self._data_path = Path(dataset_dir)
        self.metadata = pd.read_csv(self._data_path / "HAM10000_metadata.csv")
        self.metadata = self.metadata.loc[start:start+count].reset_index(drop=True)
        self._start = start
        self._count = count
        self._transform = transform
        assert 0 <= start, "Invalid start value"
        assert start + count <= self.metadata.shape[0], "Invalid count"

        labels = self.metadata.dx.unique()
        self.label_map = dict(zip(labels, range(len(labels))))
        self._part1_folder = self._data_path / "HAM10000_images_part_1"
        self._part2_folder = self._data_path / "HAM10000_images_part_2"

    def __len__(self):
        return self._count

    def __getitem__(self, idx):
        img_id, label = self.metadata.loc[idx, ["image_id", "dx"]]
        name = f"{img_id}.jpg"
        path1 = self._part1_folder / name
        path2 = self._part2_folder / name
        valid_path = path1 if path1.exists() else path2

        image = read_image(str(valid_path))
        if self._transform:
            image = self._transform(image)
        return image, self.label_map[label]


class Explorer:
    """A class helps to explore and visualize the dataset"""
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
        n_rows = ceil(len(data) / 3)
        plt.figure(figsize=(9, 4 * n_rows))
        for i, (label, image) in enumerate(zip(data, images)):
            plt.subplot(n_rows, 3, i + 1)
            image = image.permute(1, 2, 0)
            plt.imshow(image)
            plt.title(label)

    def _load_images(self, ids):
        images = []
        for idx in ids:
            image, _ = self.dataset[idx]
            images.append(image)
        return images
