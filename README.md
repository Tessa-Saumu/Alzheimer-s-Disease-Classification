# Alzheimer's MRI: an experiment I could not trust

I compared three image-classification models on MRI scans, then investigated why the results looked unusually good. The important outcome was withdrawing the accuracy claim after finding that related images crossed the evaluation splits.

This is a data-integrity case study, not an Alzheimer's diagnostic tool.

## What I did

I implemented the notebook experiments, image-intensity normalization, comparisons of how much of each pretrained model to retrain, model selection and saved prediction files. I then documented the duplicate investigation and the reasons not to deploy the result.

The images are third-party data; I did not collect or clinically label them. The original work was uploaded in one bulk commit, so the repository does not preserve an incremental development trail. The later portfolio cleanup added a small executable duplicate audit and clearer limits; it did not retrain the models.

## What the results mean

The experiment split individual images, not patients. Similar and pre-augmented images could therefore appear on both sides of the split. A model doing well in that setting has not demonstrated performance on unseen patients.

| Model | Historical test accuracy | Interpretation |
|---|---:|---|
| DenseNet121 | 99.83% | Withdrawn as evidence of generalization |
| ResNet50 | 99.71% | Same contaminated evaluation |
| EfficientNetB0 | 25.45% | Predicted one class after checkpoint reload; cause unresolved |

Each saved test file contains **6,600 rows**. Recalculating accuracy from those CSV files confirmed the recorded values; it does not validate the split. EfficientNet's 25.45% describes the final reloaded checkpoint, not every experiment. There is no controlled evidence that its failure was caused by batch normalization or my image normalization.

![Historical confusion matrix from the rejected DenseNet evaluation](results_and_visuals/densenet/confusion_matrix.png)

## Run the small audit

The new script accepts a CSV listing image `path,split`, with paths relative to that CSV and split labels `train`, `validation` or `test`. Use only images you have permission to process.

```bash
python -m pip install -r requirements-audit.txt
python -m pytest tests -q
python audit_duplicates.py permitted_split.csv --output audit_result.json
```

**Two tests passed** using images created locally for the tests. They check exact duplicates across train/test, a visually similar but byte-distinct false match, and invalid manifest entries. No medical images are required for those checks.

The output records the manifest and image SHA256 hashes, which fingerprint file contents, alongside a simple visual average-hash check. Exact file matches and visual matches are different kinds of evidence. A visual match is a reason to investigate; it does not identify a patient. The synthetic checks verify the script's mechanics, not the original dataset's leakage rate.

## What remains unresolved

The image source, redistribution licence and patient identities are not established. The exact Kaggle split used for training and the trained weights are not included. Private input paths and an unpinned training environment prevent a verified training rerun. There is no clean patient-level test, external validation, clinical deployment or demonstrated clinical benefit.

The [original report](notebooks/Alzheimer%27s_Disease_Final_Project_Report.ipynb) retains the historical screenshot investigation and hypotheses. The [saved results](results_and_visuals/) and [DenseNet](notebooks/Alzheimer%27s_densenet-experiments-and-final-training.ipynb), [ResNet](notebooks/Alzheimer%27s_resnet-experiments-and-final-training.ipynb) and [EfficientNet](notebooks/Alzheimer%27s_efficientnet-experiments-and-final-training.ipynb) notebooks retain the experiment record. See [data provenance and audit limits](DATA_PROVENANCE.md) before processing images.


[Theresia Saumu on LinkedIn](https://www.linkedin.com/in/theresia-saumu/)
