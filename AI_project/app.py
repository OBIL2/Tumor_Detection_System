# Obil Nathaniel : 271048001
# Inderias Samson : 271052262
# INSTALL: pip install tensorflow==2.20.0 streamlit numpy Pillow
# RUN:     streamlit run app.py

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import time

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide"
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH           = "neuro_model.h5"
IMG_SIZE             = (224, 224)
CLASS_NAMES          = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]
CONFIDENCE_THRESHOLD = 0.60

CLASS_INFO = {
    "Glioma": {
        "icon":  "🔴",
        "desc":  "A tumor originating from glial cells. Requires immediate specialist review.",
        "badge": "TUMOR DETECTED",
    },
    "Meningioma": {
        "icon":  "🟠",
        "desc":  "A tumor arising from the meninges. Usually slow-growing but needs evaluation.",
        "badge": "TUMOR DETECTED",
    },
    "No Tumor": {
        "icon":  "🟢",
        "desc":  "No abnormal mass detected in this scan. Results appear normal.",
        "badge": "CLEAR",
    },
    "Pituitary Tumor": {
        "icon":  "🔵",
        "desc":  "A tumor in the pituitary gland region. Specialist consultation recommended.",
        "badge": "TUMOR DETECTED",
    },
}

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file '{MODEL_PATH}' not found. Place it in the same folder as app.py.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def looks_like_mri(image: Image.Image) -> bool:
    img_rgb = image.convert("RGB").resize((64, 64))
    arr     = np.array(img_rgb, dtype=np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    return np.std([r.mean(), g.mean(), b.mean()]) < 30.0


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 NeuroScan AI")
    st.markdown("**Brain Tumor Classification**")
    st.divider()

    st.markdown("### About")
    st.info(
        "This system uses a **VGG16** deep learning model trained on **7,200 brain MRI scans** "
        "to classify tumors into 4 categories."
    )

    st.markdown("### Classification Classes")
    st.markdown("🔴 &nbsp; **Glioma** — Class 01")
    st.markdown("🟠 &nbsp; **Meningioma** — Class 02")
    st.markdown("🟢 &nbsp; **No Tumor** — Class 03")
    st.markdown("🔵 &nbsp; **Pituitary Tumor** — Class 04")

    st.divider()
    st.markdown("### Model Specs")
    st.markdown("- Architecture: **VGG16**")
    st.markdown("- Input size: **224 × 224 px**")
    st.markdown("- Classes: **4**")
    st.markdown("- Training set: **5,600 images**")
    st.markdown("- Test set: **1,600 images**")

    st.divider()
    st.caption("Obil Nathaniel · 271048001")
    st.caption("Inderias Samson · 271052262")
    st.caption("⚠️ Academic use only. Not a medical device.")


# ── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 NeuroScan AI")
st.markdown("#### Brain Tumor MRI Classification System")
st.divider()

# ── Top metrics row ───────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("🏗️ Architecture",   "VGG16")
m2.metric("🖼️ Training Images", "7,200")
m3.metric("🎯 Classes",         "4")
m4.metric("📐 Input Size",      "224 × 224")

st.divider()

# ── Upload section ────────────────────────────────────────────────────────────
st.markdown("### 📤 Upload MRI Scan")
st.markdown("Upload a brain MRI image and the model will classify it instantly.")

uploaded_file = st.file_uploader(
    "Drag and drop or click to browse",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supported: JPG, PNG, BMP, WEBP — Expected: Axial / Coronal / Sagittal MRI"
)

# ── Analysis ──────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.divider()
    st.markdown("### 🔬 Analysis")

    col_scan, col_result = st.columns([1, 1.4], gap="large")

    with col_scan:
        st.markdown("**📷 Uploaded Scan**")
        st.image(image, use_container_width=True)
        st.caption(f"📏 {image.size[0]} × {image.size[1]} px &nbsp;|&nbsp; Mode: {image.mode} &nbsp;|&nbsp; {uploaded_file.name}")

    with col_result:
        st.markdown("**🤖 AI Diagnostic Result**")

        # ── Invalid image ─────────────────────────────────────────────────
        if not looks_like_mri(image):
            st.error(
                "### ⚠️ Invalid Input\n\n"
                "This image does not appear to be a brain MRI scan. "
                "The model detected unusual colour patterns inconsistent with MRI data.\n\n"
                "**Please upload a valid grayscale brain MRI image.**"
            )

        else:
            # ── Progress animation ────────────────────────────────────────
            steps = [
                "Initialising neural pipeline",
                "Preprocessing image",
                "Extracting VGG16 features",
                "Running classifier",
                "Computing probabilities",
                "Done",
            ]
            bar    = st.progress(0, text="Starting analysis...")
            for i, step in enumerate(steps):
                pct = int((i + 1) / len(steps) * 100)
                bar.progress(pct, text=f"⬡ {step}...")
                time.sleep(0.2)
            bar.empty()

            # ── Predict ───────────────────────────────────────────────────
            scores          = model.predict(preprocess(image), verbose=0)[0]
            best_idx        = int(np.argmax(scores))
            confidence      = float(scores[best_idx])
            predicted_class = CLASS_NAMES[best_idx]
            info            = CLASS_INFO[predicted_class]
            is_safe         = predicted_class == "No Tumor"

            # ── Low confidence ────────────────────────────────────────────
            if confidence < CONFIDENCE_THRESHOLD:
                st.warning(
                    f"### ⚠️ Low Confidence\n\n"
                    f"Best match: **{predicted_class}** at **{confidence*100:.1f}%** confidence.\n\n"
                    f"This is below the minimum threshold of **60%**. "
                    f"The scan may be unclear or not a standard MRI view. "
                    f"Please upload a higher quality image."
                )

            else:
                # ── Result ────────────────────────────────────────────────
                result_text = (
                    f"### {info['icon']} {predicted_class}\n\n"
                    f"**Confidence: {confidence*100:.2f}%** &nbsp;|&nbsp; **{info['badge']}**\n\n"
                    f"{info['desc']}"
                )
                if is_safe:
                    st.success(result_text)
                else:
                    st.error(result_text)

            # ── Probability breakdown ─────────────────────────────────────
            st.markdown("---")
            st.markdown("**📊 Probability Breakdown**")

            for i, (cls, prob) in enumerate(zip(CLASS_NAMES, scores)):
                inf   = CLASS_INFO[cls]
                is_b  = (i == best_idx)
                col_l, col_r = st.columns([4, 1])
                with col_l:
                    label = f"{inf['icon']} **{cls}**" if is_b else f"{inf['icon']} {cls}"
                    st.markdown(("→ " if is_b else "&nbsp;&nbsp;&nbsp;") + label)
                    st.progress(float(prob))
                with col_r:
                    st.markdown(f"**{prob*100:.1f}%**" if is_b else f"{prob*100:.1f}%")
                    st.markdown("")   # spacer

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.info(
        "👆 **Upload an MRI scan above to get started.**\n\n"
        "The model will analyse the image and return a classification result "
        "with confidence scores for all four categories."
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🧠 NeuroScan AI &nbsp;·&nbsp; VGG16 Transfer Learning &nbsp;·&nbsp; "
    "Obil Nathaniel (271048001) &nbsp;·&nbsp; Inderias Samson (271052262) &nbsp;·&nbsp; "
    "⚠️ For academic use only — not a certified medical device"
)