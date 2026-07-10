"""
preprocess.py

Handles everything related to reading the raw dataset off disk, computing
data-quality signals, building a leakage-safe train/val/test split, and the
tf.data input pipeline used for training/evaluation.
"""

import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --- Constants ---
BASE_DIR = "../data/chest_xray"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

LABEL_MAP = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}

CLASS_NAMES = {
    0: "NORMAL",
    1: "PNEUMONIA"
}


def build_dataframe():
    """Walk the train/val/test folders and build a single dataframe of image paths + labels.

    The original Kaggle 'val' split only has 16 images, which is too small to act
    as a real validation set on its own. It's pooled here with 'train' and a
    proper stratified split is created downstream in `split_dataset`.
    """
    records = []

    for dataset_name, folder in [("Train", TRAIN_DIR), ("Val", VAL_DIR), ("Test", TEST_DIR)]:
        for label in ["NORMAL", "PNEUMONIA"]:
            class_dir = os.path.join(folder, label)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    records.append({
                        "Path": os.path.join(class_dir, fname),
                        "Label": label,
                        "Dataset": dataset_name
                    })

    return pd.DataFrame(records)


def compute_image_stats(df):
    """Per-image metadata (dimensions, brightness, contrast) for EDA / error analysis.

    Keeps `Path` in the output so any image (e.g. a misclassified one) can be
    traced back to the exact file it came from.
    """
    image_stats = []

    for _, row in df.iterrows():
        image = cv2.imread(row["Path"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        height, width = image.shape

        image_stats.append({
            "Dataset": row["Dataset"],
            "Label": row["Label"],
            "Path": row["Path"],
            "Width": width,
            "Height": height,
            "Aspect_Ratio": width / height,
            "Brightness": image.mean(),
            "Contrast": image.std()
        })

    return pd.DataFrame(image_stats)


def compute_quality_metrics(paths):
    """Blur (Laplacian variance) and entropy for a list of image paths."""
    blur_scores = []
    entropy_scores = []

    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            blur_scores.append(np.nan)
            entropy_scores.append(np.nan)
            continue

        blur = cv2.Laplacian(image, cv2.CV_64F).var()

        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))

        blur_scores.append(blur)
        entropy_scores.append(entropy)

    return blur_scores, entropy_scores


def split_dataset(df):
    """Re-split the pooled dataset into a stratified train/val/test.

    Test set is carved out first and is never touched again until final
    evaluation - not for training, validation, early stopping, or threshold
    tuning. This keeps the final test metrics an honest generalization
    estimate rather than a number that's been indirectly tuned against.

    Note: this public dataset doesn't expose patient IDs, so it's possible
    the same patient's images appear in more than one split. That's a
    limitation of the dataset itself, not something fixable here.
    """
    df = df.copy()
    df["Target"] = df["Label"].map(LABEL_MAP)

    train_val_df = df[df["Dataset"].isin(["Train", "Val"])]
    test_df = df[df["Dataset"] == "Test"].copy()

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.15,
        stratify=train_val_df["Target"],
        random_state=SEED
    )

    return train_df.copy(), val_df.copy(), test_df


def compute_class_weights(train_df):
    """Balanced class weights to counter the PNEUMONIA/NORMAL imbalance.

    Re-weighting the loss (instead of oversampling/undersampling) avoids
    duplicating samples or throwing away real training data.
    """
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["Target"]),
        y=train_df["Target"]
    )
    return dict(enumerate(class_weights))


def load_image(path, label):
    """Read + decode + resize a single image.

    Uses `decode_image` (not `decode_jpeg`) because the dataset can contain
    both .jpeg and .png files - `decode_jpeg` throws on real PNG bytes.
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


def build_datasets(train_df, val_df, test_df, batch_size=BATCH_SIZE):
    """Build the train/val/test tf.data pipelines used for training and evaluation."""

    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices((train_df["Path"].values, train_df["Target"].values))
        .map(load_image, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(1024, seed=SEED)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    val_dataset = (
        tf.data.Dataset
        .from_tensor_slices((val_df["Path"].values, val_df["Target"].values))
        .map(load_image, num_parallel_calls=AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    test_dataset = (
        tf.data.Dataset
        .from_tensor_slices((test_df["Path"].values, test_df["Target"].values))
        .map(load_image, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train_dataset, val_dataset, test_dataset


def get_processed_data(batch_size=BATCH_SIZE):
    """Convenience entry point: raw folders -> split dataframes -> tf.data datasets + class weights."""
    df = build_dataframe()
    train_df, val_df, test_df = split_dataset(df)
    class_weight_dict = compute_class_weights(train_df)
    train_dataset, val_dataset, test_dataset = build_datasets(train_df, val_df, test_df, batch_size)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "class_weight_dict": class_weight_dict,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset
    }


if __name__ == "__main__":
    data = get_processed_data()
    print(f"Train: {len(data['train_df'])} | Val: {len(data['val_df'])} | Test: {len(data['test_df'])}")
    print("Class weights:", data["class_weight_dict"])