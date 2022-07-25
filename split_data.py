import os
from pathlib import Path

import pandas as pd


def get_images_fps(imgs_root):
    img_df = pd.DataFrame(
        {"img_fp": list(map(str, Path(imgs_root).rglob("*.jpg")))},
    )
    img_df["is_dog"] = img_df["img_fp"].map(
        lambda fp: int("dog" in os.path.basename(fp).lower())
    )
    return img_df


if __name__ == "__main__":
    dataset_root = "cats_vs_dogs"
    train_root = os.path.join(dataset_root, "train")
    test_root = os.path.join(dataset_root, "test")

    splits_dest_dir = "data_splits"
    os.makedirs(splits_dest_dir, exist_ok=True)

    df_train = get_images_fps(train_root)
    df_test_split = get_images_fps(test_root)
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    size = len(df_train)
    train_size = int(0.8 * size)
    df_train_split = df_train.iloc[:train_size]
    df_validation_split = df_train.iloc[train_size:]

    df_train_split.to_csv(os.path.join(splits_dest_dir, "train.csv"))
    df_validation_split.to_csv(os.path.join(splits_dest_dir, "validation.csv"))
    df_test_split.to_csv(os.path.join(splits_dest_dir, "test.csv"))
