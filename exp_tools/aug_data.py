from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import itertools
import os

import pandas as pd
from torchvision.io import read_image, write_jpeg
import torchvision.transforms.v2 as tfs

from exp_tools.data_utils import HAM10000_LABEL_MAP

HAM10000_CLASSES = list(HAM10000_LABEL_MAP.keys())

aug_pipeline = tfs.Compose(
    [
        tfs.RandomRotation(180, fill=(0,)),
        tfs.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=(0,)),
        tfs.RandomHorizontalFlip(p=0.5),
        tfs.RandomVerticalFlip(p=0.5),
        tfs.Resize((224, 224)),
    ]
)


def collect_image_paths(image_ids, image_dirs):
    """Collects and returns image paths from image_dirs."""
    paths = []
    for img_id in image_ids:
        for d in image_dirs:
            path = os.path.join(d, f"{img_id}.jpg")
            if os.path.exists(path):
                paths.append(path)
                break
    return paths


def transform_and_save(args):
    """Applies the appropriate transforms on the source images and
    saves them to the desired path."""
    transform, source_path, dest_path = args
    img = read_image(source_path)
    img = transform(img)
    # img = (img.clamp(0, 1) * 255).byte()
    write_jpeg(img.cpu(), dest_path, quality=90)


def copy_image(args):
    """Simply copies images from one path to another."""
    source_path, dest_path = args
    img = read_image(source_path)
    write_jpeg(img, dest_path)


def generate_augmented_data_ham10000(
    source_dir,
    dest_dir,
    ratio=0.8,
    num_samples_per_class=8000,
    transform=aug_pipeline,
    num_workers=8,
):
    os.makedirs(dest_dir, exist_ok=True)
    train_dir = os.path.join(dest_dir, "train")
    test_dir = os.path.join(dest_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    data = pd.read_csv(os.path.join(source_dir, "HAM10000_metadata.csv"))
    img_dirs = [
        os.path.join(source_dir, "HAM10000_images_part_1"),
        os.path.join(source_dir, "HAM10000_images_part_2"),
    ]
    print("Loaded image directories and metadata....✅")
    with (
        ProcessPoolExecutor(max_workers=num_workers) as proc_exec,
        ThreadPoolExecutor(max_workers=num_workers) as thread_exec,
    ):
        for cls in HAM10000_CLASSES:
            samples = data[data.dx == cls]
            dest_train_dir = os.path.join(train_dir, cls)
            dest_test_dir = os.path.join(test_dir, cls)
            os.makedirs(dest_train_dir, exist_ok=True)
            os.makedirs(dest_test_dir, exist_ok=True)

            train_end = int(samples.shape[0] * ratio)
            train_samples = samples.iloc[:train_end]
            test_samples = samples.iloc[train_end:]

            # --- Train augmentation ---
            base_paths = collect_image_paths(train_samples.image_id, img_dirs)
            path_gen = itertools.cycle(base_paths)
            source_paths = [next(path_gen) for _ in range(num_samples_per_class)]
            dest_paths = [
                os.path.join(dest_train_dir, f"{i:05d}.jpg")
                for i in range(num_samples_per_class)
            ]

            proc_exec.map(
                transform_and_save,
                [(transform, s, d) for s, d in zip(source_paths, dest_paths)],
            )

            # --- Test copying ---
            test_source_paths = collect_image_paths(test_samples.image_id, img_dirs)
            test_dest_paths = [
                os.path.join(dest_test_dir, f"{i:04d}.jpg")
                for i in range(len(test_source_paths))
            ]
            thread_exec.map(copy_image, zip(test_source_paths, test_dest_paths))

            print(
                f"[{cls}] train: {len(source_paths)}, test: {len(test_source_paths)} augmentation started..."
            )

    print("✅ Augmentation complete.")
