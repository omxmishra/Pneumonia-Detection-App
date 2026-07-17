"""
train.py

Trains the custom CNN and benchmarks it against transfer-learning backbones,
fine-tunes the best-performing one, evaluates it on the held-out test set,
and saves the final model + metrics.

Run directly:
    python train.py
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from preprocess import get_processed_data, compute_image_stats, CLASS_NAMES
from model import build_model, unfreeze_for_finetuning

MODELS_DIR = "../models"
EPOCHS_HEAD = 20
EPOCHS_FINETUNE = 10
UNFREEZE_LAYERS = 30
FINETUNE_LR = 1e-5

os.makedirs(MODELS_DIR, exist_ok=True)


def get_callbacks(model_name):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODELS_DIR, f"{model_name}.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    return [early_stopping, reduce_lr, checkpoint]


def train_model(model, model_name, train_dataset, val_dataset, class_weight_dict, epochs=EPOCHS_HEAD):
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=get_callbacks(model_name),
        verbose=1
    )
    return history


def evaluate_model(model, dataset):
    """Runs predictions on a dataset and returns metrics + raw predictions."""
    y_true = np.concatenate([labels.numpy() for _, labels in dataset])
    y_prob = model.predict(dataset, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob)
    }

    return metrics, y_true, y_pred, y_prob


def benchmark_backbones(train_dataset, val_dataset, test_dataset, class_weight_dict):
    """Trains the custom CNN + all registered transfer-learning backbones and compares them."""
    results = []
    trained_models = {}
    backbones = {}
    histories = {}

    for model_name in ["CNN", "MobileNetV2", "DenseNet121", "EfficientNetB0"]:
        print(f"\n{'=' * 60}\nTraining: {model_name}\n{'=' * 60}")

        model, backbone = build_model(model_name)
        history = train_model(model, model_name, train_dataset, val_dataset, class_weight_dict)

        metrics, y_true, y_pred, y_prob = evaluate_model(model, test_dataset)
        metrics["Model"] = model_name

        trained_models[model_name] = model
        backbones[model_name] = backbone
        histories[model_name] = history
        results.append(metrics)

    results_df = pd.DataFrame(results).set_index("Model").sort_values("Accuracy", ascending=False)
    return results_df, trained_models, backbones, histories


def run_error_analysis(test_df, y_true, y_pred, y_prob):
    """Cross-checks misclassified images against blur/brightness/contrast to
    distinguish genuine model weaknesses from errors on low-quality scans."""
    import cv2

    test_reset = test_df.reset_index(drop=True)
    misclassified_mask = (y_true != y_pred)
    misclassified_df = test_reset[misclassified_mask].copy()
    misclassified_df["True_Label"] = misclassified_df["Target"].map(CLASS_NAMES)
    misclassified_df["Pred_Prob"] = y_prob[misclassified_mask]

    quality_rows = []
    for path in misclassified_df["Path"]:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        blur = cv2.Laplacian(image, cv2.CV_64F).var()
        quality_rows.append({
            "Brightness": image.mean(),
            "Contrast": image.std(),
            "Blur_Score": blur
        })

    misclassified_df = misclassified_df.reset_index(drop=True).join(pd.DataFrame(quality_rows))

    print(f"Total misclassified: {len(misclassified_df)} / {len(test_reset)} test images")
    return misclassified_df


def main():
    print("Loading and preprocessing data...")
    data = get_processed_data()

    print("\nBenchmarking architectures...")
    results_df, trained_models, backbones, histories = benchmark_backbones(
        data["train_dataset"], data["val_dataset"], data["test_dataset"], data["class_weight_dict"]
    )
    print("\nBenchmark results:")
    print(results_df)

    best_model_name = results_df.index[0]
    print(f"\nBest model: {best_model_name}")

    best_model = trained_models[best_model_name]
    base_model = backbones[best_model_name]

    if base_model is not None:
        print(f"\nFine-tuning {best_model_name} (unfreezing last {UNFREEZE_LAYERS} layers)...")
        best_model = unfreeze_for_finetuning(best_model, base_model, UNFREEZE_LAYERS, FINETUNE_LR)
        train_model(
            best_model, f"{best_model_name}_finetuned",
            data["train_dataset"], data["val_dataset"], data["class_weight_dict"],
            epochs=EPOCHS_FINETUNE
        )

    print("\nFinal evaluation on held-out test set...")
    final_metrics, y_true, y_pred, y_prob = evaluate_model(best_model, data["test_dataset"])
    print(final_metrics)
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    run_error_analysis(data["test_df"], y_true, y_pred, y_prob)

    save_path = os.path.join(MODELS_DIR, f"pneumonia_{best_model_name.lower()}_finetuned.keras")
    best_model.save(save_path)
    results_df.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"))
    print(f"\nSaved final model to {save_path}")


if __name__ == "__main__":
    main()