"""
predict.py

Run inference on a single chest X-ray image using a saved model.

CLI usage:
    python predict.py --image path/to/xray.jpeg --model ../models/pneumonia_densenet121_finetuned.keras
    python predict.py --image path/to/xray.jpeg --model ../models/pneumonia_densenet121_finetuned.keras --gradcam
"""

import argparse

import cv2
import numpy as np
import tensorflow as tf

from preprocess import IMG_SIZE, CLASS_NAMES
from model import generate_gradcam, overlay_gradcam, get_last_conv_layer


def load_and_preprocess(image_path):
    """Read + decode + resize a single image for inference (matches training pipeline)."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image


def predict_image(model, image_path):
    """Predict NORMAL/PNEUMONIA for a single image, returning label + confidence."""
    image = load_and_preprocess(image_path)
    image_batch = tf.expand_dims(image, axis=0)

    probability = model.predict(image_batch, verbose=0)[0][0]
    prediction = int(probability >= 0.5)
    confidence = probability if prediction == 1 else 1 - probability

    return {
        "Prediction": CLASS_NAMES[prediction],
        "Confidence": float(confidence),
        "Probability": float(probability)
    }


def predict_with_gradcam(model, image_path, output_path=None):
    """Predict + generate a Grad-CAM overlay saved to disk (or returned as an array)."""
    image = load_and_preprocess(image_path)

    result = predict_image(model, image_path)

    last_conv_layer_name = get_last_conv_layer(model)
    heatmap = generate_gradcam(model, image, last_conv_layer_name)

    raw_image_uint8 = np.uint8(image.numpy())
    overlay = overlay_gradcam(raw_image_uint8, heatmap, IMG_SIZE)

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    result["gradcam_overlay"] = overlay
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict NORMAL/PNEUMONIA from a chest X-ray image.")
    parser.add_argument("--image", required=True, help="Path to the chest X-ray image.")
    parser.add_argument("--model", required=True, help="Path to the saved .keras model.")
    parser.add_argument("--gradcam", action="store_true", help="Also generate a Grad-CAM overlay.")
    parser.add_argument("--output", default="gradcam_overlay.png", help="Where to save the Grad-CAM overlay.")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)

    if args.gradcam:
        result = predict_with_gradcam(model, args.image, args.output)
        print(f"Prediction : {result['Prediction']}")
        print(f"Confidence : {result['Confidence']:.4f}")
        print(f"Grad-CAM overlay saved to: {args.output}")
    else:
        result = predict_image(model, args.image)
        print(f"Prediction : {result['Prediction']}")
        print(f"Confidence : {result['Confidence']:.4f}")


if __name__ == "__main__":
    main()