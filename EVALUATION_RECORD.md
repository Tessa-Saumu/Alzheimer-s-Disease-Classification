# Historical evaluation record

These scores are retained for traceability, not as evidence of unseen-patient or clinical performance. No new training or evaluation was performed for the 16 October documentation update.

| Model | Historical test accuracy | Meaning |
|---|---:|---|
| DenseNet121 | 99.83% | Withdrawn as evidence of generalization |
| ResNet50 | 99.71% | Same contaminated evaluation |
| EfficientNetB0 | 25.45% | Final reloaded checkpoint only; cause unresolved |

Each saved test CSV contains **6,600 rows**. Recalculation confirmed these values, without validating the split. **Two synthetic audit tests passed** on 13 October 2026, checking exact cross-split duplicates, a visual false match and invalid manifests. No medical images were required or reprocessed in that check.
