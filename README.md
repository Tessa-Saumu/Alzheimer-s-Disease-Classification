# Alzheimer's Disease Classification: Architecture Search & Data Integrity Audit

## Project Overview
This project focuses on the multi-class classification of Alzheimer’s Disease stages using MRI scans. What began as a standard transfer learning implementation (**Version 1**) evolved into a rigorous investigation of model architecture, preprocessing strategies, and dataset integrity (**Version 2**).

While the final model (**DenseNet121**) achieved a test accuracy of **99.83%**, a post-training forensic audit revealed critical flaws in the open-source dataset (data leakage via pre-augmentation). This project demonstrates the importance of rigorous MLOps practices, showing that **data validation is just as critical as model architecture.**

---

## Evolution: Version 1 vs. Version 2

This project represents a significant leap in engineering maturity and methodology.

| Feature | **Version 1 (Baseline)** | **Version 2 (Advanced Pipeline)** |
| :--- | :--- | :--- |
| **Model** | ResNet50 (Random Unfreezing) | **DenseNet121** (Block-Wise Fine-Tuning) |
| **Preprocessing** | Standard Rescaling (1./255) | **Per-Image IQR Normalization** (Medical Standard) |
| **Data Split** | Random Split | **Stratified Split** (Preventing Class Imbalance) |
| **Selection Logic** | Best Validation Accuracy | **Minimizing Generalization Gap** (Train vs. Val) |
| **Outcome** | 89% Accuracy (High False Positives) | **99.8% Accuracy** (Triggered Data Audit) |

---

## Methodology (V2)

### 1. Robust Preprocessing Strategy
Unlike natural images (ImageNet), MRI scans vary significantly in brightness depending on the scanner hardware.
*   **The Solution:** implemented **Per-Image Robust Normalization** using the Median and Interquartile Range (IQR).
*   **The Result:** This removed scanner artifacts and forced the model to learn structural brain atrophy rather than pixel intensity, solving the "Covariate Shift" issue that caused models like EfficientNet to fail.

### 2. Architecture Search & Block-Wise Fine-Tuning
Instead of unfreezing arbitrary layers, I utilized a **Block-Wise Unfreezing** strategy to respect the hierarchical nature of CNNs (Lines $\to$ Shapes $\to$ Structures).
*   **Models Tested:** ResNet50, EfficientNetB0, DenseNet121.
*   **Winner:** **DenseNet121** (Freezing first 140 layers). Its feature concatenation architecture proved most robust to the medical data distribution.

---

## Experimental Results

| Model | Configuration | Test Accuracy | F1-Score (Macro) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **DenseNet121** | Freeze 140 Layers | **99.83%** | **1.00** | **Converged** |
| **ResNet50** | Unfreeze All | 99.71% | 0.99 | Runner Up |
| **EfficientNetB0**| All Configs | 25.45% | 0.20 | Failed (Distribution Mismatch) |

### Performance on Key Classes
The V2 model resolved the clinical trade-offs found in V1, specifically regarding the confusion between **NonDemented** and **VeryMildDemented**.

| Class | Precision | Recall | Support |
| :--- | :--- | :--- | :--- |
| **MildDemented** | 1.00 | 1.00 | 1500 |
| **ModerateDemented** | 1.00 | 1.00 | 1500 |
| **NonDemented** | 1.00 | 1.00 | 1920 |
| **VeryMildDemented** | 1.00 | 1.00 | 1680 |

---

## The Forensic Audit: "Too Good To Be True?"

Achieving 99.83% accuracy in medical imaging is statistically improbable and often indicates **Data Leakage**. I refused to accept the results at face value and conducted a forensic audit of the dataset.

### The Findings
1.  **Duplicate Images:** The dataset contained identical images rotated slightly, existing in both Train and Test sets.
2.  **Pre-Augmentation:** The source data was heavily augmented *before* release, breaking the independence of the test set.
3.  **Conclusion:** The model was successfully memorizing the augmentations rather than generalizing to new patients.

### Critical Lesson
**Data Quality > Model Sophistication.** While the pipeline (Preprocessing + Architecture Search) works perfectly, the results cannot be clinically deployed until the model is retrained on a pristine, non-leaky dataset (e.g., ADNI).

---

## Clinical Implications & Roadmap

Despite the dataset flaw, the **Recall optimization strategy** remains valid for future iterations:

1.  **Immediate Screening Assistant:** The architecture is tuned to prioritize **Recall**, ensuring no severe cases (`ModerateDemented`) are missed.
2.  **Algorithm Validity:** The success of **DenseNet121 + IQR Normalization** over EfficientNet establishes a clear technical path for processing grayscale medical scans.
3.  **Next Steps:**
    *   Secure raw, non-augmented data (ADNI/OASIS).
    *   Retrain the current pipeline on the clean data.
    *   Implement 3D Volumetric analysis for deeper insight.

---

## Technical Stack
*   **Deep Learning:** TensorFlow, Keras, Transfer Learning (DenseNet/ResNet)
*   **Data Processing:** NumPy, Pandas, Scikit-Learn (Stratified Splitting)
*   **Visualization:** Matplotlib, Seaborn
*   **Environment:** Google Colab and Kaggle (T4 GPU)

## Contact
**Tessa Saumu**
*   **LinkedIn:** [www.linkedin.com/in/theresia-saumu-2a1910307](www.linkedin.com/in/theresia-saumu-2a1910307)
*   **Email:** [theresia.saumu@gmail.com](mailto:theresia.saumu@gmail.com)
