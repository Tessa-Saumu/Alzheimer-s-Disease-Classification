# Supporting audit and implementation plan (2026-10-13)

Source: remote main 2038dac7f8c94441b1202b317e37bfb0ee0b3c5b; SUPPORTING failure case, never a diagnostic product. Latest remote docs already withdraw clinical/generalization meaning of 99.83% and remove the unsupported EfficientNet root-cause story; retain those prior changes.

## Findings and completed repairs

Added executable manifest-driven exact/perceptual hash audit, pinned lightweight audit requirements, provenance note and synthetic tests. Two tests pass: exact train/test contamination, perceptual false-match example, rejected split labels and repeated paths. Sparse checkout excluded dataset_images; no medical images downloaded/reprocessed by this repair. The code saves manifest SHA256 and image-level hashes/split labels. It does not recover original consumed split or measure original leakage percentages.

## Acceptance and next steps

1. Supporting written retrospective and synthetic audit mechanics are ready after review. Saved historical prediction artifacts can be inspected; training reproduction is not established.
2. To claim a reproduced original audit, first resolve image source/licence and exact consumed split manifest, then run this script on permitted images and archive manifest/output together. Until then historical contamination findings are attributed to historical evidence, not this run.
3. Patient-level validation, clean retraining and checkpoint-collapse investigation are separate large research work, deferred. Do not publish weights as evidence of validity.

Safe positioning: identified and withdrew unreliable image-level accuracy; packaged an inspectable contamination audit with limits. Hash equality is not patient identity; provenance and clinical validity remain unresolved.

## Current verification checkpoint

2 synthetic audit tests pass; historical three CSV accuracies/row counts recomputed. No original image/split audit or training reproduced.
