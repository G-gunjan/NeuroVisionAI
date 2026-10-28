# Brain MRI AI Clinical Platform

**Four-Class Brain Tumor Classification** with Fine-Tuned EfficientNetB0, Grad-CAM Explainability, RBM Latent Features & VAE Image Synthesis


> **Research prototype for decision support only — not a medical device and not intended for clinical diagnosis.**

---

## Overview

End-to-end system that classifies brain MRI slices into four classes:

- **Glioma**
- **Meningioma**
- **Pituitary tumor**
- **No tumor**

Built with a fine-tuned **EfficientNetB0**, Grad-CAM visual explanations, entropy-based out-of-distribution detection, optional recall-bias mode, Restricted Boltzmann Machine (RBM) latent features, and a Variational Autoencoder (VAE) for synthetic MRI generation. Everything is packaged in a Streamlit web application.

### Key Results (Test Set)

| Metric              | Value   |
|---------------------|---------|
| Accuracy            | 88.50%  |
| Macro F1-score      | 0.88    |
| Glioma Recall       | 0.73    |
| Meningioma Recall   | 0.84    |
| No-tumor Recall     | 0.99    |
| Pituitary Recall    | 0.98    |

Validation accuracy: **92.26%**

---

## Features

- Fine-tuned EfficientNetB0 (top 45 layers trainable) on 128×128 MRI slices
- Class-weighted training (higher penalty for missed gliomas & meningiomas)
- Grad-CAM heatmaps for explainability
- Entropy-based Out-of-Distribution (OOD) warning
- Adjustable confidence / uncertainty threshold
- Optional recall-bias mode at inference
- 128-D CNN features + 64-D RBM latent representation (linear probe)
- Convolutional VAE for synthetic MRI generation
- Clean Streamlit interface with Diagnostic, Generator, and Benchmarks tabs

---

## Dataset

- **7,200** balanced brain MRI images (1,800 per class)
- Split: 4,760 train / 840 validation / 1,600 test
- Images resized to **128×128**, EfficientNet preprocessing applied

---
Model Details
Backbone: EfficientNetB0 (ImageNet pretrained)
Input: 128×128×3
Head: GlobalAveragePooling → Dense(128) → BatchNorm → Dropout(0.4) → Softmax(4)
Optimizer: Adam (1e-4)
Class weights: Glioma ×1.8, Meningioma ×1.3, others ×0.8
Augmentation: Horizontal flip, rotation, zoom, contrast (inside model)
Auxiliary Models
RBM: 128 → 64 latent features (CD-1)
VAE: 32-D latent space, grayscale 128×128

Novelty & Contributions
Clinically motivated class weighting + optional inference-time recall bias
Built-in reliability layer (entropy OOD + confidence threshold)
Grad-CAM working correctly with nested EfficientNet models
Unified study of CNN features vs RBM compression
Honest benchmarking + complete data-to-deployment package
Limitations
Glioma recall (0.73) is still too low for safety-critical use
Single train/val/test split (no cross-validation or patient-level split)
128×128 resolution loses fine detail
VAE is unconditional (class selector in UI does not currently condition generation)
Research prototype only — not for clinical diagnosis
Future Work
Higher resolution (224×224+) and stronger backbones
Patient-level splits + cross-validation
Focal loss / threshold tuning for glioma
Conditional VAE or diffusion models
External dataset validation & calibrated OOD detector
DisclaimerThis project is a research and educational prototype. It is not a certified medical device and must not be used for clinical diagnosis or treatment decisions. Always consult qualified medical professionals.References
Tan & Le – EfficientNet (ICML 2019)
Selvaraju et al. – Grad-CAM (ICCV 2017)
Kingma & Welling – Auto-Encoding Variational Bayes (ICLR 2014)
Hinton – Contrastive Divergence (Neural Computation 2002)
Krizhevsky et al. – ImageNet Classification (NeurIPS 2012)
