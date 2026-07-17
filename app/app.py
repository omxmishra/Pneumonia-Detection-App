"""
app.py

Streamlit app for the Pneumonia Chest X-Ray Classifier.
Upload a chest X-ray, get a NORMAL/PNEUMONIA prediction with confidence,
a Grad-CAM heatmap, and a look at how the model was built and benchmarked.

Run (from inside app/):
    streamlit run app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import IMG_SIZE, CLASS_NAMES
from model import generate_gradcam, get_last_conv_layer, overlay_gradcam

MODEL_PATH = "../models/pneumonia_densenet121_finetuned.keras"

# Actual benchmark results from training (see ../models/model_comparison.csv)
BENCHMARK_RESULTS = pd.DataFrame({
    "Model": ["DenseNet121", "EfficientNetB0", "MobileNetV2", "CNN (custom)"],
    "Accuracy": [0.867, 0.864, 0.841, 0.699],
    "Precision": [0.853, 0.849, 0.806, 0.675],
    "Recall": [0.951, 0.951, 0.982, 0.997],
    "AUC": [0.950, 0.939, 0.959, 0.916],
}).set_index("Model")

FINAL_METRICS = {
    "Accuracy": 0.880,
    "Precision": 0.874,
    "Recall": 0.944,
    "F1": 0.908,
    "AUC": 0.959
}

st.set_page_config(
    page_title="Pneumonia X-Ray Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------
# Global styling
# ------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 20% 0%, #12151c 0%, #0a0c10 55%, #08090c 100%);
}

#MainMenu, footer, header {visibility: hidden;}

/* Hero */
.hero {
    padding: 2.2rem 2.4rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #151922 0%, #0f1218 100%);
    border: 1px solid #232733;
    margin-bottom: 1.6rem;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f2f3f5;
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}
.hero-sub {
    color: #8b93a3;
    font-size: 0.98rem;
    line-height: 1.5;
}
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    background: #1c2531;
    color: #6fd7a3;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid #2b3a4a;
    margin-right: 0.5rem;
    margin-top: 0.8rem;
}

/* Cards */
.card {
    background: #12151c;
    border: 1px solid #212633;
    border-radius: 16px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 1rem;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #151922 0%, #101318 100%);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    border: 1px solid #232733;
    margin-top: 0.5rem;
}
.result-label {
    font-size: 1.05rem;
    color: #8b93a3;
    margin-bottom: 0.3rem;
}
.result-value-pneumonia {
    font-size: 2.1rem;
    font-weight: 800;
    color: #f16565;
}
.result-value-normal {
    font-size: 2.1rem;
    font-weight: 800;
    color: #5fd68f;
}
.conf-bar-bg {
    width: 100%;
    height: 10px;
    background: #1c202a;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 0.6rem;
}
.conf-bar-fill-pneumonia {
    height: 100%;
    background: linear-gradient(90deg, #f16565, #ff8a80);
    border-radius: 999px;
}
.conf-bar-fill-normal {
    height: 100%;
    background: linear-gradient(90deg, #37b874, #5fd68f);
    border-radius: 999px;
}

/* Metric chips */
.metric-chip {
    background: #12151c;
    border: 1px solid #212633;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-chip-label {
    color: #78808f;
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.metric-chip-value {
    color: #f2f3f5;
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 0.15rem;
}

/* Uploader */
[data-testid="stFileUploader"] {
    background: #12151c;
    border: 1.5px dashed #2b3140;
    border-radius: 14px;
    padding: 0.6rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: #12151c;
    border-radius: 10px 10px 0 0;
    color: #8b93a3;
    padding: 0.6rem 1.2rem;
    border: 1px solid #212633;
}
.stTabs [aria-selected="true"] {
    background: #1a1e28 !important;
    color: #f2f3f5 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0c0e13;
    border-right: 1px solid #1c202a;
}

hr {
    border-color: #1c202a;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🫁 Model Snapshot")
    st.caption("DenseNet121, fine-tuned")

    for label, value in FINAL_METRICS.items():
        st.markdown(
            f"""
            <div class="metric-chip" style="margin-bottom:0.5rem;">
                <div class="metric-chip-label">{label}</div>
                <div class="metric-chip-value">{value:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.caption("Test set: 624 held-out chest X-rays, never used in training or tuning.")
    st.caption("Built with TensorFlow/Keras · Grad-CAM explainability")


# ------------------------------------------------------------------------
# Hero header
# ------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">Pneumonia Chest X-Ray Classifier</div>
    <div class="hero-sub">
        Upload a chest X-ray and get a NORMAL / PNEUMONIA prediction, backed by a
        fine-tuned DenseNet121 and a Grad-CAM heatmap showing what the model
        actually looked at.
    </div>
    <span class="badge">● DenseNet121</span>
    <span class="badge">Test Accuracy 88.0%</span>
    <span class="badge">Recall (Pneumonia) 94.4%</span>
</div>
""", unsafe_allow_html=True)

st.warning("Educational / portfolio project — not a diagnostic tool. Do not use for real medical decisions.", icon="⚠️")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    return np.array(image).astype("float32")


# ------------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------------
tab_predict, tab_performance, tab_about = st.tabs(["🔍  Predict", "📊  Model Performance", "ℹ️  About"])

with tab_predict:
    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.markdown("#### Upload an X-ray")
        uploaded_file = st.file_uploader(
            "Drop a chest X-ray image (JPEG / PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        show_gradcam = st.toggle("Show Grad-CAM attention heatmap", value=True)

        if uploaded_file is not None:
            st.image(
                Image.open(uploaded_file).convert("RGB"),
                caption="Uploaded X-ray",
                use_container_width=True
            )

    with right:
        if uploaded_file is not None:
            model = load_model()
            image_array = preprocess_uploaded_image(uploaded_file)
            image_tensor = tf.convert_to_tensor(image_array)

            with st.spinner("Running inference..."):
                image_batch = tf.expand_dims(image_tensor, axis=0)
                probability = float(model.predict(image_batch, verbose=0)[0][0])
                prediction = int(probability >= 0.5)
                confidence = probability if prediction == 1 else 1 - probability

            label = CLASS_NAMES[prediction]
            is_pneumonia = prediction == 1
            value_class = "result-value-pneumonia" if is_pneumonia else "result-value-normal"
            bar_class = "conf-bar-fill-pneumonia" if is_pneumonia else "conf-bar-fill-normal"

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Prediction</div>
                    <div class="{value_class}">{label}</div>
                    <div class="result-label" style="margin-top:1rem;">Confidence: {confidence:.1%}</div>
                    <div class="conf-bar-bg">
                        <div class="{bar_class}" style="width:{confidence*100:.1f}%;"></div>
                    </div>
                    <div class="result-label" style="margin-top:1rem;">
                        Raw probability of PNEUMONIA: <b>{probability:.4f}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if show_gradcam:
                st.markdown("#### Grad-CAM: where the model looked")
                last_conv_layer_name = get_last_conv_layer(model)
                heatmap = generate_gradcam(model, image_tensor, last_conv_layer_name)
                overlay = overlay_gradcam(image_array.astype("uint8"), heatmap, IMG_SIZE)
                st.image(overlay, use_container_width=True)
                st.caption("Warmer regions influenced the prediction more.")
        else:
            st.markdown(
                """
                <div class="card" style="text-align:center; color:#5c6472; padding: 3.5rem 1rem;">
                    Upload an X-ray on the left to see a prediction here.
                </div>
                """,
                unsafe_allow_html=True
            )

with tab_performance:
    st.markdown("#### Architecture benchmark (before fine-tuning)")
    st.dataframe(
        BENCHMARK_RESULTS.style.format("{:.1%}").highlight_max(axis=0, color="#1c3a2a"),
        use_container_width=True
    )
    st.caption(
        "DenseNet121 was selected for fine-tuning based on the best overall accuracy/AUC "
        "trade-off, then had its last 30 layers unfrozen at a low learning rate."
    )

    st.markdown("#### Final model, held-out test set (624 images)")
    cols = st.columns(len(FINAL_METRICS))
    for col, (label, value) in zip(cols, FINAL_METRICS.items()):
        with col:
            st.markdown(
                f"""
                <div class="metric-chip">
                    <div class="metric-chip-label">{label}</div>
                    <div class="metric-chip-value">{value:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("")
    st.info(
        "Recall on PNEUMONIA (94.4%) is prioritized over raw accuracy — a missed "
        "diagnosis (false negative) is costlier than a false alarm in a screening "
        "context. The model catches most PNEUMONIA cases but is comparatively more "
        "conservative on NORMAL scans (77% class recall), erring toward flagging "
        "borderline cases rather than clearing them outright.",
        icon="📌"
    )

with tab_about:
    st.markdown("#### How this was built")
    st.markdown("""
- **Data:** Kaggle Chest X-Ray Images (Pneumonia) dataset — re-split with a stratified,
  leakage-aware train/val/test partition (test set isolated until final evaluation).
- **Class imbalance:** handled with `class_weight="balanced"` during training, rather
  than oversampling or discarding data.
- **Model selection:** a custom CNN and three transfer-learning backbones
  (MobileNetV2, DenseNet121, EfficientNetB0) were benchmarked under identical
  conditions before picking a winner.
- **Fine-tuning:** the last 30 layers of DenseNet121 were unfrozen at a low learning
  rate, with BatchNorm layers kept frozen to protect pretrained running statistics.
- **Explainability:** Grad-CAM highlights the regions of the X-ray driving each
  prediction, as a sanity check that the model is learning lung-relevant features.
    """)
    st.markdown("#### Limitations")
    st.markdown("""
- This dataset doesn't expose patient IDs, so the same patient's images could
  appear across multiple splits — a limitation of the public dataset, not the code.
- Trained and evaluated on X-rays from a single source; performance on scans from
  a different hospital or imaging equipment is untested.
    """)


if __name__ == "__main__":
    pass