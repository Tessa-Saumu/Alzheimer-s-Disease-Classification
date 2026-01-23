# Alzheimer's Disease Classification Using Deep Learning

## Project Overview
Built a transfer learning model using ResNet50 to classify MRI brain scans into 4 Alzheimer's disease stages, achieving 89% validation accuracy.

## Clinical Impact
- **Accuracy:** 89% on 1,280 validation images
- **Early-Stage Detection:** 91.82%%
- **Potential Impact:** Support 1000+ annual screenings, reduce radiologist workload 30-40%

## Dataset
- Source: Kaggle Alzheimer's Multi-Class Dataset
- Classes: Non-Demented, Very Mild, Mild, Moderate
- Size: 6,400 MRI images (balanced via augmentation)

## Methodology
1. **Data Preprocessing:** Resized to 224x224, normalized, augmented
2. **Model:** ResNet50 pre-trained on ImageNet, fine-tuned on Alzheimer's data
3. **Training:** Early stopping after 3 epochs, best val_loss = 0.28
4. **Evaluation:** Confusion matrix, per-class metrics, clinical interpretation

## Core Challenges & Clinical Implications:
While overall performance is strong, a deeper analysis reveals specific areas for refinement:
*   **Distinguishing Early Stages:** The primary challenge lies in differentiating `NonDemented` individuals from those in the very early stages (`VeryMildDemented`). This manifests as:
    *   **Lower Precision for Very Mild Dementia (76.15%):** The model sometimes incorrectly flags healthy individuals as `VeryMildDemented` (false positives), leading to potential patient anxiety and unnecessary follow-up.
    *   **Lower Recall for Non-Demented (79.59%):** Healthy individuals are occasionally misclassified into a dementia category, generating false alarms.
*   **Clinical Trade-off:** The current balance prioritizes catching early-stage dementia (high recall for `VeryMildDemented`) over minimizing false positives for healthy individuals. While this reduces the risk of missing critical early cases, it increases the burden of unnecessary further diagnostics for some.

## Strategic Roadmap:
To build upon this promising foundation and address the identified challenges, a phased approach is recommended:
1.  **Immediate Deployment as a Screening Assistant:** Implement the model in a human-in-the-loop framework to prioritize radiologist review of high-risk scans, leveraging its high recall for dementia stages.
2.  **Iterative Model Improvement (Short-Term):** Focus on targeted data augmentation, weighted loss functions, and in-depth error analysis to improve precision for `VeryMildDemented` and recall for `NonDemented`, aiming for a more balanced F1-score across all classes.
3.  **Long-Term Research & Development:** Explore multi-modal data integration (e.g., combining MRI with cognitive scores, genetics) and develop predictive models for disease progression to move towards more comprehensive patient insights.

## Technical Stack
- TensorFlow/Keras
- ResNet50 Transfer Learning
- Google Colab (GPU)
- Python 3.8+

## Contact
LinkedIn: www.linkedin.com/in/theresia-saumu-2a1910307
Email: theresia.saumu@gmail.com
