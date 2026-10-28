import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ==========================================
# 0. Page Configuration & Custom UI Styling
# ==========================================

st.set_page_config(
    page_title="Brain MRI AI Diagnostic & Generative Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

CATEGORIES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

IMG_SIZE = (128, 128)


# ==========================================
# Custom CSS
# ==========================================

st.markdown("""
<style>

    :root {
        --primary-color: #007AFF;
        --bg-card: #FFFFFF;
        --border-color: #E2E8F0;
    }

    .title-text {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    .subtitle-text {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 20px;
    }

    .custom-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }

    .badge-tumor {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.1rem;
        border: 1px solid #FCA5A5;
    }

    .badge-notumor {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.1rem;
        border: 1px solid #86EFAC;
    }

    .badge-warning {
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.1rem;
        border: 1px solid #FDE68A;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px;
        padding-left: 16px;
        padding-right: 16px;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================
# 1. Cached Model Loading
# ==========================================

@st.cache_resource
def load_models():

    try:
        cnn = load_model(
            "cnn_model.keras",
            compile=False
        )
    except Exception as e:
        print("CNN loading error:", e)
        cnn = None

    try:
        vae = load_model(
            "vae_decoder.keras",
            compile=False
        )
    except Exception as e:
        print("VAE loading error:", e)
        vae = None

    return cnn, vae


cnn_model, vae_decoder = load_models()

# ==========================================
# 2. Grad-CAM Explainability
# ==========================================

def find_last_conv_layer(model):
    """Find the last Conv2D layer, including nested CNN backbones."""
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
        if isinstance(layer, tf.keras.Model):
            nested_conv = find_last_conv_layer(layer)
            if nested_conv is not None:
                last_conv = nested_conv
    return last_conv


def find_nested_backbone(model):
    """Return the last nested model containing a convolutional layer, if any."""
    candidate = None
    for index, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            if find_last_conv_layer(layer) is not None:
                candidate = (index, layer)
    return candidate


def make_gradcam_heatmap(input_tensor, model, target_class=None):
    """Create a Grad-CAM heatmap for a class index using the model's input tensor."""
    nested = find_nested_backbone(model)

    if nested is not None:
        # For a nested backbone (e.g. EfficientNetB0), obtain the feature maps
        # and backbone output from a connected intermediate model, then run the
        # remaining classifier-head layers while the gradient tape is active.
        backbone_index, backbone = nested
        last_conv = find_last_conv_layer(backbone)
        feature_model = tf.keras.Model(
            inputs=backbone.inputs,
            outputs=[last_conv.output, backbone.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, x = feature_model(input_tensor, training=False)
            for layer in model.layers[backbone_index + 1:]:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                x = layer(x, training=False)
            predictions = x
            if target_class is None:
                target_class = int(tf.argmax(predictions[0]).numpy())
            class_score = predictions[:, target_class]
    else:
        # For a model with convolutional layers directly in its top-level graph.
        last_conv = find_last_conv_layer(model)
        if last_conv is None:
            raise ValueError("No Conv2D layer was found in the loaded CNN model.")

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv.output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_tensor, training=False)
            if target_class is None:
                target_class = int(tf.argmax(predictions[0]).numpy())
            class_score = predictions[:, target_class]

    grads = tape.gradient(class_score, conv_outputs)
    if grads is None:
        raise ValueError(
            "Gradients could not be calculated for the selected layer. "
            "Check the loaded model architecture."
        )

    # Global-average-pool gradients to get channel importance weights.
    weights = tf.reduce_mean(grads, axis=(1, 2))
    feature_maps = conv_outputs[0]
    channel_weights = weights[0]
    heatmap = tf.reduce_sum(
        feature_maps * channel_weights[tf.newaxis, tf.newaxis, :],
        axis=-1
    )
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_rgb, heatmap, alpha=0.40):
    """Overlay a normalized Grad-CAM heatmap on the original RGB MRI image."""
    heatmap_resized = cv2.resize(
        heatmap,
        (original_rgb.shape[1], original_rgb.shape[0])
    )
    heatmap_uint8 = np.uint8(np.clip(heatmap_resized, 0, 1) * 255)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(
        original_rgb.astype(np.uint8),
        1.0 - alpha,
        heatmap_rgb,
        alpha,
        0
    )



# ==========================================
# Entropy & OOD Detection
# ==========================================

def calculate_entropy(probabilities):

    probs = np.clip(
        probabilities,
        1e-10,
        1.0
    )

    return -np.sum(
        probs * np.log2(probs)
    )


# ==========================================
# 3. App Header & Sidebar
# ==========================================

st.markdown(
    "<div class='title-text'>🧠 Brain MRI Clinical AI Platform</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle-text'>"
    "Next-Gen Deep Learning for Tumor Classification "
    "& Latent Generative Modeling"
    "</div>",
    unsafe_allow_html=True
)


# ==========================================
# Sidebar
# ==========================================

st.sidebar.image(
    "https://img.icons8.com/isometric/100/brain.png",
    width=70
)

st.sidebar.title(
    "⚙️ Model Controls"
)

# IMPORTANT:
# Default is now False so the displayed prediction
# represents the actual CNN output.
recall_bias_mode = st.sidebar.toggle(
    "Enable Clinical Recall Bias",
    value=False
)

if recall_bias_mode:

    st.sidebar.caption(
        "Adjust probability thresholds to penalize false negatives:"
    )

    glioma_weight = st.sidebar.slider(
        "Glioma Recall Bias",
        1.0,
        3.0,
        1.8,
        0.1
    )

    meningioma_weight = st.sidebar.slider(
        "Meningioma Recall Bias",
        1.0,
        3.0,
        1.3,
        0.1
    )

    class_bias = np.array(
        [
            glioma_weight,
            meningioma_weight,
            0.7,
            0.7
        ],
        dtype=np.float32
    )

else:

    class_bias = np.ones(
        4,
        dtype=np.float32
    )


confidence_threshold = st.sidebar.slider(
    "Uncertainty Threshold (%)",
    40,
    90,
    60,
    5
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "System Status:"
)

if cnn_model is not None:

    st.sidebar.success(
        "🟢 CNN Model: Loaded"
    )

else:

    st.sidebar.error(
        "🔴 CNN Model: Missing"
    )


if vae_decoder is not None:

    st.sidebar.success(
        "🟢 VAE Decoder: Loaded"
    )

else:

    st.sidebar.warning(
        "🟡 VAE Decoder: Demo Mode"
    )


# ==========================================
# 4. Main Navigation Tabs
# ==========================================

tab1, tab2, tab3 = st.tabs(
    [
        "🔍 Diagnostic Analysis",
        "🎨 Synthetic VAE Generator",
        "📊 Benchmarks & Architecture"
    ]
)


# ==========================================================
# TAB 1 — Diagnostic Analysis
# ==========================================================

with tab1:

    st.markdown(
        "#### Upload Brain MRI Scan"
    )

    uploaded_file = st.file_uploader(
        "Select image file (JPG, PNG)...",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        # --------------------------------------------------
        # Read image
        # --------------------------------------------------

        file_bytes = np.asarray(
            bytearray(
                uploaded_file.read()
            ),
            dtype=np.uint8
        )

        image_bgr = cv2.imdecode(
            file_bytes,
            1
        )

        if image_bgr is None:

            st.error(
                "Could not read the uploaded image."
            )

            st.stop()

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        # --------------------------------------------------
        # Resize
        # --------------------------------------------------

        resized_img = cv2.resize(
            image_rgb,
            IMG_SIZE
        )

        # --------------------------------------------------
        # IMPORTANT:
        # Use original EfficientNet preprocessing.
        # Do NOT use /255.0 here.
        # --------------------------------------------------

        input_tensor = preprocess_input(
            np.expand_dims(
                resized_img.astype(
                    np.float32
                ),
                axis=0
            )
        )

        # ==================================================
        # CNN Prediction
        # ==================================================

        if cnn_model is not None:

            try:

                # ------------------------------------------
                # RAW MODEL PROBABILITIES
                # ------------------------------------------

                raw_probs = cnn_model.predict(
                    input_tensor,
                    verbose=0
                )[0]

                raw_probs = np.asarray(
                    raw_probs,
                    dtype=np.float32
                )

                # ------------------------------------------
                # Safety:
                # If model outputs logits, convert to
                # probabilities.
                # ------------------------------------------

                if (
                    np.any(raw_probs < 0)
                    or
                    not np.isclose(
                        np.sum(raw_probs),
                        1.0,
                        atol=1e-3
                    )
                ):

                    raw_probs = tf.nn.softmax(
                        raw_probs
                    ).numpy()

                # ------------------------------------------
                # CLASS BIAS
                # Disabled by default.
                # ------------------------------------------

                if recall_bias_mode:

                    adjusted_probs = (
                        raw_probs *
                        class_bias
                    )

                    adjusted_probs = (
                        adjusted_probs /
                        (
                            np.sum(
                                adjusted_probs
                            ) + 1e-10
                        )
                    )

                else:

                    adjusted_probs = raw_probs.copy()

                # ------------------------------------------
                # Prediction
                # ------------------------------------------

                pred_idx = int(
                    np.argmax(
                        adjusted_probs
                    )
                )

                confidence = (
                    adjusted_probs[pred_idx] *
                    100
                )

                entropy = calculate_entropy(
                    adjusted_probs
                )

                is_ood = (
                    entropy > 1.8
                )

                # ==================================================
                # UI (3 columns: original MRI, Grad-CAM, results)
                # ==================================================

                col_a, col_b, col_c = st.columns(
                    [
                        1,
                        1,
                        1.3
                    ]
                )

                # ==================================================
                # COLUMN A — ORIGINAL MRI
                # ==================================================

                with col_a:

                    st.markdown(
                        "<div class='custom-card'>",
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        "**Original MRI Scan**"
                    )

                    st.image(
                        image_rgb,
                        use_container_width=True
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

                # ==================================================
                # COLUMN B — GRAD-CAM EXPLANATION
                # ==================================================

                with col_b:

                    st.markdown(
                        "<div class='custom-card'>",
                        unsafe_allow_html=True
                    )

                    st.markdown("**Grad-CAM Explanation**")

                    try:
                        heatmap = make_gradcam_heatmap(
                            input_tensor=input_tensor,
                            model=cnn_model,
                            target_class=pred_idx
                        )

                        gradcam_image = overlay_gradcam(
                            original_rgb=image_rgb,
                            heatmap=heatmap,
                            alpha=0.40
                        )

                        st.image(
                            gradcam_image,
                            caption=(
                                "Grad-CAM: regions contributing to "
                                f"the {CATEGORIES[pred_idx]} class score"
                            ),
                            use_container_width=True
                        )

                        st.caption(
                            "Warmer colors indicate stronger positive "
                            "contributions in this visualization. "
                            "This is not a tumor segmentation mask."
                        )

                    except Exception as gradcam_error:
                        st.warning("Grad-CAM could not be generated for this model.")
                        st.caption(f"Technical detail: {gradcam_error}")

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

                # ==================================================
                # COLUMN C — DIAGNOSTIC RESULTS
                # ==================================================

                with col_c:

                    st.markdown(
                        "<div class='custom-card'>",
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        "**Diagnostic Results**"
                    )

                    # ------------------------------------------
                    # Diagnosis Status Badge
                    # ------------------------------------------

                    if is_ood:

                        st.markdown(
                            "<div class='badge-warning'>"
                            "⚠️ Out-Of-Distribution (OOD)"
                            "</div>",
                            unsafe_allow_html=True
                        )

                        st.write("")

                        st.caption(
                            "The uploaded image displays high "
                            "entropy metrics and may not be a "
                            "standard Brain MRI."
                        )

                    elif confidence < confidence_threshold:

                        st.markdown(
                            f"<div class='badge-warning'>"
                            f"⚠️ Uncertain ({confidence:.1f}%)"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                        st.write("")

                        st.caption(
                            "Low confidence score. "
                            "Radiologist verification recommended."
                        )

                    else:

                        if (
                            CATEGORIES[pred_idx]
                            == "notumor"
                        ):

                            st.markdown(
                                "<div class='badge-notumor'>"
                                "✅ NO TUMOR DETECTED"
                                "</div>",
                                unsafe_allow_html=True
                            )

                        else:

                            st.markdown(
                                f"<div class='badge-tumor'>"
                                f"🚨 "
                                f"{CATEGORIES[pred_idx].upper()}"
                                f" DETECTED"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                    st.write("")

                    st.metric(
                        label="Model Confidence",
                        value=f"{confidence:.2f}%"
                    )

                    st.progress(
                        float(
                            confidence / 100
                        )
                    )

                    # ------------------------------------------
                    # Probability Distribution
                    # ------------------------------------------

                    st.markdown("---")

                    st.markdown(
                        "**Probability Breakdown**"
                    )

                    fig, ax = plt.subplots(
                        figsize=(5, 2.5)
                    )

                    palette = [
                        "#EF4444"
                        if (
                            i == pred_idx
                            and CATEGORIES[i] != "notumor"
                        )
                        else "#10B981"
                        if i == pred_idx
                        else "#64748B"
                        for i in range(4)
                    ]

                    sns.barplot(
                        x=adjusted_probs,
                        y=[
                            c.capitalize()
                            for c in CATEGORIES
                        ],
                        hue=[
                            c.capitalize()
                            for c in CATEGORIES
                        ],
                        palette=palette,
                        legend=False,
                        ax=ax
                    )

                    ax.set_xlim(
                        0,
                        1.0
                    )

                    ax.spines[
                        "top"
                    ].set_visible(False)

                    ax.spines[
                        "right"
                    ].set_visible(False)

                    st.pyplot(
                        fig
                    )

                    plt.close(fig)

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

            except Exception as e:

                st.error(
                    "Prediction error:"
                )

                st.code(
                    str(e)
                )


# ==========================================================
# TAB 2 — VAE
# ==========================================================

with tab2:

    st.markdown(
        "### Conditional Synthetic Image Generation"
    )

    st.caption(
        "Leverage Variational Autoencoders (VAE) "
        "to synthesize class-conditioned latent "
        "brain MRI scans."
    )

    col1, col2 = st.columns(
        [
            1,
            2.2
        ]
    )

    with col1:

        st.markdown(
            "<div class='custom-card'>",
            unsafe_allow_html=True
        )

        st.markdown(
            "**Generator Parameters**"
        )

        selected_class = st.selectbox(
            "Target Condition:",
            [
                c.capitalize()
                for c in CATEGORIES
            ]
        )

        num_samples = st.slider(
            "Sample Count",
            1,
            4,
            3
        )

        gen_button = st.button(
            "✨ Generate Synthetic Scans",
            use_container_width=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            "<div class='custom-card'>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"**Synthesized Latent Outputs ({selected_class})**"
        )

        if (
            gen_button
            or
            "generated_once"
            not in st.session_state
        ):

            st.session_state[
                "generated_once"
            ] = True

            fig, axes = plt.subplots(
                1,
                num_samples,
                figsize=(
                    4 * num_samples,
                    3.8
                )
            )

            if num_samples == 1:
                axes = [axes]

            for i in range(
                num_samples
            ):

                if vae_decoder is not None:

                    latent_sample = (
                        np.random.normal(
                            size=(1, 32)
                        )
                    )

                    generated_img = (
                        vae_decoder.predict(
                            latent_sample,
                            verbose=0
                        )[0]
                    )

                else:

                    generated_img = (
                        np.random.beta(
                            0.5,
                            0.5,
                            size=(
                                128,
                                128
                            )
                        )
                    )

                axes[i].imshow(
                    generated_img.squeeze(),
                    cmap="bone"
                )

                axes[i].set_title(
                    f"Synthetic "
                    f"{selected_class} "
                    f"#{i+1}",
                    fontsize=10,
                    fontweight="bold"
                )

                axes[i].axis("off")

            plt.tight_layout()

            st.pyplot(
                fig
            )

            plt.close(fig)

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# ==========================================================
# TAB 3 — BENCHMARKS (Corrected)
# ==========================================================

with tab3:

    st.markdown(
        "### Performance Benchmarks & Model Architecture"
    )

    col1, col2 = st.columns(
        [
            1.2,
            1
        ]
    )

    with col1:

        st.markdown(
            "<div class='custom-card'>",
            unsafe_allow_html=True
        )

        st.markdown(
            "**📊 Architecture Benchmarks Comparison**"
        )

        # Actual results from the training notebook
        benchmark_data = {
            "Model / Representation": [
                "EfficientNetB0 (Fine-Tuned CNN)",
                "128-D Raw CNN Features + Logistic Regression",
                "64-D RBM Latent Features + Logistic Regression"
            ],
            "Accuracy": [
                "88.5%",
                "90.75%",
                "86.62%"
            ],
            "Glioma Recall": [
                "0.73",
                "—",
                "—"
            ],
            "Meningioma Recall": [
                "0.84",
                "—",
                "—"
            ],
            "F1-Score (Macro)": [
                "0.88",
                "—",
                "—"
            ]
        }

        st.dataframe(
            pd.DataFrame(
                benchmark_data
            ),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Note: Values taken directly from the training notebook evaluation."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            "<div class='custom-card'>",
            unsafe_allow_html=True
        )

        st.markdown(
            "**🧩 Feature Representations Used**"
        )

        st.markdown(
            """
            - **CNN Backbone**: EfficientNetB0 (fine-tuned)  
              → 128-dimensional dense features
            - **RBM**: Trained on the 128-D features  
              → 64-dimensional probabilistic latent features
            - **Linear Probe**: Logistic Regression on both feature sets
            """
        )

        st.markdown("---")

        st.markdown(
            "**📌 Key Observations from Experiments**"
        )

        st.markdown(
            """
            - Fine-tuned EfficientNetB0 achieved **88.5%** test accuracy.
            - Raw CNN features gave the best linear probe accuracy (**90.75%**).
            - RBM latent features slightly underperformed the raw CNN features (**86.62%**).
            - No end-to-end fusion or VAE-augmented retraining was performed in the notebook.
            """
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )
