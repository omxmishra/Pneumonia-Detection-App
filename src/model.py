"""
model.py

Model architecture definitions (custom CNN + transfer-learning backbones)
and Grad-CAM utilities for explainability.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    BatchNormalization,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2, DenseNet121, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

from preprocess import IMG_SIZE, SEED

# --- Backbones available for benchmarking ---
MODELS = {
    "MobileNetV2": {
        "backbone": MobileNetV2,
        "preprocess": mobilenet_preprocess
    },
    "DenseNet121": {
        "backbone": DenseNet121,
        "preprocess": densenet_preprocess
    },
    "EfficientNetB0": {
        "backbone": EfficientNetB0,
        "preprocess": efficientnet_preprocess
    }
}

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
], name="data_augmentation")


def build_model(model_name):
    """Build either the custom CNN or a transfer-learning model.

    Returns (model, backbone). The backbone reference is returned directly
    because backbones built with `input_tensor=x` get flattened into the
    outer model's layer list - there is no nested `Model` layer to search for
    afterwards, so this reference must be kept at build time (needed later
    for fine-tuning).
    """
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    backbone = None

    if model_name == "CNN":
        x = layers.Rescaling(1. / 255)(inputs)

        x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)

        x = GlobalAveragePooling2D()(x)

    else:
        if model_name not in MODELS:
            raise ValueError(f"Unknown model_name '{model_name}'. Choose from: CNN, {list(MODELS.keys())}")

        x = data_augmentation(inputs)
        x = MODELS[model_name]["preprocess"](x)

        backbone = MODELS[model_name]["backbone"](
            include_top=False,
            weights="imagenet",
            input_tensor=x
        )
        backbone.trainable = False

        x = GlobalAveragePooling2D()(backbone.output)

    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    return model, backbone


def unfreeze_for_finetuning(model, backbone, num_layers=30, learning_rate=1e-5):
    """Unfreeze the last `num_layers` of the backbone for fine-tuning.

    BatchNormalization layers within that range are explicitly re-frozen -
    fine-tuning with a small batch size would otherwise corrupt their
    pretrained running mean/variance statistics and destabilize training.
    """
    backbone.trainable = True

    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False

    for layer in backbone.layers[-num_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    return model


# --- Grad-CAM ---

def get_last_conv_layer(model):
    """Find the last Conv2D layer in the model's flattened layer graph."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def generate_gradcam(model, image, last_conv_layer_name=None):
    """Generate a Grad-CAM heatmap (values in [0, 1]) for a single preprocessed image."""
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    image_batch = tf.expand_dims(image, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_gradcam(raw_image, heatmap, img_size, alpha=0.4):
    """Overlay a Grad-CAM heatmap on top of the original (uint8, HxWx3) image."""
    heatmap_resized = cv2.resize(heatmap, (img_size, img_size))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    base_image = np.uint8(raw_image)
    overlay = cv2.addWeighted(base_image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay