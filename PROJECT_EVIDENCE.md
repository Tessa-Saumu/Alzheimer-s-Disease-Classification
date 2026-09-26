# Project Evidence — Alzheimer's Disease Classification (Multi-Class CNN with Block-Wise Fine-Tuning & Data-Leakage Audit)

> Forensic documentation of repository `Tessa-Saumu/Alzheimer-s-Disease-Classification` as of commit `97bec18e` (the only commit in history).
> Every claim below is tagged as **[verified]** (proven by repository evidence), **[documented-not-reproduced]** (stated in docs but not independently confirmable), **[incomplete]**, **[abandoned/broken]**, or **[undeterminable]**.

---

## Context

**When:**
- GitHub repository created 2026-01-23 (repo metadata), single push/commit dated **2026-05-26**.
- Notebook execution evidence: V2 experiments and final training all ran on **2026-02-01** (`results_and_visuals/*/experiment_config.json` and `main_config.json` timestamps, 12:27–15:26 UTC, ~3 hours of GPU time); TensorFlow startup logs in notebooks show the same date.
- The dataset-leakage audit evidence is a screenshot dated **2026-02-02** (filename `Screenshot 2026-02-02 012548.png` embedded in `notebooks/Alzheimer's_Disease_Final_Project_Report.ipynb`).
- V1 (initial project) execution date is **[undeterminable]** — no preserved timestamps.
- **[verified]** via commit metadata, config JSONs, notebook logs.

**Why it existed:**
- **[documented-not-reproduced]** As stated in the README and `Alzheimer's_Disease_Final_Project_Report.ipynb` (Cell 0–2): a second iteration ("Version 2") of a medical-imaging classification project, intended to fix V1's methodology (arbitrary unfreezing, plain rescaling, random split) and later to investigate why accuracy was suspiciously high. Nothing in the repository contradicts this, but the repo cannot independently prove intent.

**Solo / team:**
- **[verified]** All git authorship resolves to one identity: commit `97bec18e` by "Theresia (Tessa) Saumu <theresia.saumu@gmail.com>". README contact: Tessa Saumu. GitHub shows **0 pull requests, 0 issues, 0 forks, 0 stars, no other branches** (checked via `gh`).
- **Caveat:** the entire repository (495 MB, 44,000 images, all notebooks) landed in Git as **a single squashed commit** titled "Update LinkedIn profile link in README". There is no development history; contribution boundaries cannot be proven from Git. Solo authorship is consistent with all available evidence but not independently provable.

**My role:**
- Per the only author identity present: sole author of notebooks, README, results artifacts, and the Kaggle/Colab executions they record. **[verified]** as sole authorship of the repository contents; the underlying work process is **[undeterminable]** from history.

---

## Problem

**What problem was actually being solved?**

- **Implemented problem [verified]:** 4-class image classification of axial brain MRI slices into `MildDemented`, `ModerateDemented`, `NonDemented`, `VeryMildDemented`, using transfer learning (ResNet50 / EfficientNetB0 / DenseNet121) on a 44,000-image pool shipped in `dataset_images/`. The engineering sub-problem actually executed was: *which pre-trained architecture + how many frozen layers + which intensity-normalization gives the best validation/test performance*, answered via systematic freeze-depth sweeps (6 DenseNet configs, 12 ResNet runs across two generations, 9 EfficientNet configs) recorded in executed notebook outputs and `results_and_visuals/*/results_final.csv`.
- **Late-stage implemented problem [verified]:** a post-training data-integrity investigation after the model hit 99.83% test accuracy. The conclusion (pre-augmented data leaking across splits) is stated in the README/Final Report and is **independently confirmed by my own re-analysis of the shipped dataset** (see Evidence section).
- **Aspirational framing (not implemented):** "Clinical Implications", "screening assistant", deployment roadmap, 3D volumetric analysis, ADNI/OASIS retraining. **[documented-not-reproduced]** — no deployment, no clinical partner, no new dataset exists in the repo.

**Real vs. aspirational:** the repository implements a Kaggle-style transfer-learning experiment workflow plus a leakage audit. It does **not** implement a clinical tool. The Final Report itself states "Do NOT deploy this model" (Cell 8) — an unusually honest and appropriate conclusion.

---

## What I personally built

Single-author repository; "ownership" below means "present in the repo under the sole author identity".

**Confirmed (code/notebooks with executed outputs):**
1. **`robust_normalize()` preprocessing function** — per-image median/IQR normalization over non-zero pixels, present verbatim in all three V2 training notebooks (`notebooks/Alzheimer's_{densenet,efficientnet,resnet}-experiments-and-final-training.ipynb`, cell with `def robust_normalize`). **[verified]**
2. **Freeze-depth experiment harness** — `Config` class, `create_model()`, `train_model()`, `evaluate_model()`, `run_all_experiments()` with callbacks (EarlyStopping on val_loss patience 5, ReduceLROnPlateau, ModelCheckpoint on val_accuracy), per-config CSV persistence and curve plotting. Duplicated across the three architecture notebooks. **[verified, executed]**
3. **`select_best_model()` selection algorithm** — elite-group (within 0.5% of peak val acc) → minimum |train−val| generalization gap → val-loss tiebreaker. Executed in all three notebooks. **[verified]**
4. **Final-training + test-evaluation cells** — full-epoch retraining of winner config, then test-set evaluation producing `confusion_matrix.png`, `test_predictions.csv` (6,600 rows each), `per_class_metrics.csv`. **[verified, executed]**
5. **Stratified 70/15/15 splitting script** — `split_dataset()` in `notebooks/Alzheimer's_data_pipeline_setup.ipynb` (sklearn `train_test_split`, stratified, seed 42). Present as code but **never executed in the notebook** (all `execution_count` are `None`, no outputs). The consumed split existed only as a private Kaggle dataset (`final_dementia_split`). **[incomplete-as-evidence]**
6. **EDA notebook** (`Alzheimer's_Disease_Classification_EDA_Improved.ipynb`, 14 code cells, 10 with outputs) — class counts, intensity distributions, noise/positional analysis, augmentation demo, long markdown summary. **[verified, executed on Colab]**
7. **V1 baseline notebook** (`Alzheimer's_Disease_Classification_Initial_Project_Full.ipynb`) — ResNet50 transfer learning, 80/20 split, percentage-based unfreezing; preserved training logs and classification report. **[verified, executed on Colab]**
8. **Final Project Report notebook** — 8 markdown cells, 0 code cells; narrative + embedded figures + one audit screenshot. **[verified as documentation]**
9. **README.md** — project summary. **[verified as documentation]**
10. **Results artifacts** — 3 directories of PNG/CSV/JSON outputs (`results_and_visuals/`). **[verified; cross-checked against notebook outputs — consistent]**

**Unclear ownership:**
- The dataset: provenance is **not documented anywhere in the repo** (no source URL, no citation, no license). Filename families (see Scale) strongly indicate it merges the widely circulated Kaggle "Alzheimer's Dataset – 4 classes" with an augmented expansion and further `aug_*` files. This is **inference from strong filename/count evidence, not documented fact**. **[undeterminable → treat as third-party data]**
- The pre-made Kaggle input datasets referenced in notebooks (`/kaggle/input/dataset-finals/...`, `/kaggle/input/dataset-split/...`, `/kaggle/input/dementia-splits/...`) are private/external; who created them is not provable, though timestamps and naming suggest the same author.

---

## Architecture

- **Frontend:** none. **[absent]**
- **Backend / serving:** none. No API, no inference server, no exported model in the repo (no `.h5`/`.keras`/SavedModel weights are committed — checkpoints were written to `/kaggle/working/` only). **[absent]**
- **Data:** `dataset_images/` — 44,000 JPGs in 4 class folders (**[verified]**: Mild 10,000 / Moderate 10,000 / Non 12,800 / VeryMild 11,200). Train/val/test splitting done offline; training notebooks read a pre-split tree (`final_dementia_split/{train,validation,test}`) from Kaggle via `ImageDataGenerator.flow_from_directory` with `preprocessing_function=robust_normalize`, target 224×224 RGB, batch 32. **[verified]**
- **ML/AI:** TensorFlow 2.19.0/Keras on 2× Tesla T4 via `tf.distribute.MirroredStrategy` (per preserved logs). Three ImageNet backbones with a common head: `GlobalAveragePooling2D → Dense(256, relu) → Dropout(0.5) → Dense(4, softmax)`, Adam with `clipnorm=1.0`, categorical cross-entropy. Freeze-first-N-layers strategies; LR tiered by trainable-layer ratio. Seeds fixed at 42. **[verified]**
- **Database:** none. **[absent]**
- **Infrastructure:** Google Colab (V1, EDA, pipeline) and Kaggle notebook VMs (V2 training). No Docker, no CI, no IaC, no environment lock file (no `requirements.txt`/`environment.yml` anywhere). **[absent]**

**Major data/control flow (as built):**
`dataset_images/` (44k combined pool) → offline stratified 70/15/15 split (seed 42) → private Kaggle dataset → per-architecture notebooks: freeze-depth sweep (10 epochs, 250 steps/epoch) → `select_best_model` → full final training (≤50 epochs, early stop) → best-checkpoint `.h5` → reload → test-set evaluation → CSV/PNG artifacts (later copied into repo).

---

## ML / AI work

**Problem formulation:** single-label 4-class softmax classification of 224×224×3 MRI axial slices. **[verified]**

**Target:** class label in {Mild, Moderate, Non, VeryMild}. **[verified]**; class balance near-even (22.7%–29.1%) **[verified from folder counts + notebook outputs]**.

**Features:** raw pixels → per-image median/IQR robust normalization (V2) or 1/255 rescale (V1); no hand-crafted features, no augmentation applied at training time in V2 (`ImageDataGenerator` used with preprocessing function only — **[verified]**; any augmentation was baked into the dataset itself before the author's involvement).

**Baseline:**
- V1 ResNet50 (random-percentage unfreezing, 1/255 rescale, 80/20 split, accuracy used for both checkpointing and reporting): **89.06% best val accuracy (epoch logs) / 0.89 accuracy, macro-F1 0.89 in preserved classification report** (n=8,800; VeryMild P 0.76 / R 0.92; Non R 0.80; Moderate R 1.00). **[verified from preserved outputs in V1 notebook, cells exec 17–27]**

**Models (V2 sweep, all with executed evidence):**

| Model | Configs run | Best experiment val acc (10-epoch, partial-epoch steps) | Final full training | Test accuracy |
|---|---|---|---|---|
| DenseNet121 | 6 (freeze 462/312/140/52/6/0 layers) | 0.99594 (freeze_conv3=140) | best val 0.99803 @ ep19, early stop ep24, 79.0 min | **0.99832** (11/6600 wrong) |
| ResNet50 | 6 + 6 (two generations: count-based, then block-wise; see Limitations) | 0.99344 (freeze_none) | selected freeze_none, final run | **0.99714** |
| EfficientNetB0 | 9 (freeze 237/234/221/162/119/75/46/17/0) | 0.98687 (freeze_block1=17) | final run reached val 0.99894 @ ep41, 110.6 min | **0.25455 — model collapsed to single-class prediction after checkpoint reload** |

**[verified]** — notebook logs + `results_final.csv` + `test_predictions.csv` + `per_class_metrics.csv` + confusion matrices all cross-consistent.

**Validation methodology — critically assessed:**
- Split: single stratified 70/15/15 split at **image level, with no subject/patient grouping** — on a pool that (verified below) is full of augmented near-duplicates. One fixed seed (42); no cross-validation, no subject-level holdout, no external test set. **[verified]**
- Model selection (`select_best_model`) is reasonable and is implemented/executed **[verified]**, but the "generalization gap" it minimizes is computed against a **leaked validation set**, so the gap is meaningless as a generalization estimate. The DenseNet experiments' negative gaps (val > train) are themselves a leakage symptom.
- Experiment-phase runs used `STEPS_PER_EPOCH=250` (8,000 of 30,800 train images/epoch; 100 val steps = 3,200 of 6,600) — experiment comparisons rest on partial epochs; preserved logs show repeated "input ran out of data; interrupting training" warnings. Final runs used full epochs. **[verified]**
- Checkpoint/eval integrity: final models were saved with `ModelCheckpoint(monitor=val_accuracy)` in legacy `.h5`, reloaded, then evaluated on test. EfficientNet's collapse demonstrates this save→reload→serve path was **never validated** (a model at val 0.99894 scored 0.25455 on a same-distribution test set — a train/serve skew the repo never explains). **[verified]**

**Metrics:** accuracy, per-class precision/recall/F1 (`per_class_metrics.csv`), confusion matrices, val loss, training time. **[verified]** No AUC, no calibration, no confidence intervals.

**Final result (honest statement):**
- DenseNet121 freeze_conv3: test acc **0.99832** (6,589/6,600; per-class recall 0.9969–1.0) — numerically verified in three independent repo artifacts — **but achieved on a test set with proven near-duplicate leakage (see below), so it does not measure generalization to unseen patients.** The README itself now says this. The **credible** outcome of the project is therefore not "99.8% accurate Alzheimer's classifier" but "a working transfer-learning experiment pipeline + a correctly diagnosed dataset integrity failure".

**EfficientNet root-cause claim — assessed:** README/Final Report attribute the collapse to "Covariate Shift / EfficientNet's Batch-Normalization sensitivity to custom normalization". The repo contains **no root-cause experiment** (no BN-freeze test, no val-invocation test, no checkpoint-integrity check); and the same notebook's *own experiments with the identical IQR normalization reached 0.98687 val accuracy*, which contradicts the normalization-breaks-EfficientNet narrative. **[documented-not-reproduced → insufficiently supported claim]**

---

## Engineering evidence

- **Tests:** **0 unit/integration tests in the repository** — no test files, no test framework, no CI. The only test-like code is `test_pipeline()` (a smoke-test function defined once in each V2 notebook and **never invoked** — single occurrence per notebook = definition only) and manual "data pipeline test" prints inside notebooks that did execute (batch shape/range checks). Assertions exist in the eval cell (shuffle check). **Evidence they pass:** only inline prints; no automated pass/fail record. **[verified absence + verified inline checks]**
- **APIs:** none. **[absent]**
- **CI/CD:** none (no `.github/`, no CI configs). **[absent]**
- **Containers:** none. **[absent]**
- **Architecture patterns:** Config-class pattern; duplicated experiment-harness pattern across architecture notebooks; callback-driven training; artifact persistence (CSV/JSON/PNG); selection-algorithm-as-code. **[verified]** — but heavy code duplication (three near-identical ~300-line harnesses); no shared module (`from data_pipeline_setup import ...` appears commented out in notebook cells).
- **Data validation:** EDA notebooks + the post-hoc duplicate audit (screenshot-only code, see Limitations) + my independent re-analysis (see Evidence). No pre-training validation gate existed. **[partially verified post-hoc]**
- **Error handling:** try/except around experiment loop (continue-on-failure) and data-loading test; generator checks in eval cells. **[verified as present]**
- **Monitoring:** none — print-based logs only; no TensorBoard/W&B/MLflow evidence. **[absent]**
- **Security:** no secrets/credentials found in repo (grep scan); dataset contains medical-looking images whose licensing/privacy status is **undocumented**. **[verified absence of secrets; license status undeterminable]**

---

## Scale

Only repository-supported numbers:

- 44,000 dataset images (~482 MB on disk): Mild 10,000 / Moderate 10,000 / Non 12,800 / VeryMild 11,200. **[verified]**
- Train/val/test as executed: **30,800 / 6,600 / 6,600** (notebook logs: "Found 30800 images…"). **[verified]**
- Test set per-class support: 1,500 / 1,500 / 1,920 / 1,680. **[verified]**
- Effective unique content (my audit of the shipped pool, average-hash on a stratified 6,000-image sample): **only 1,157 unique hashes; 87.1% of sampled images belong to duplicate groups.** **[verified by me, see Evidence]**
- Filename families in `dataset_images/` (my measurement): 33,984 UUID-named; 5,121 `mildDem*/moderateDem*/nonDem*/verymildDem*` originals; 1,279 `NN (k).jpg` numbered files; 3,616 `aug_NNNN_*` files (3,472 in ModerateDemented) — summing exactly to 44,000, and implying the Moderate class was bulk-augmented from ~64 originals to 10,000. **[verified by me]** Provenance mapping to specific public datasets: **[strong inference, undocumented]**.
- Originals implied: the pool derives from roughly ~6,400 base images (5,121 train + 1,279 test of the source dataset) — themselves from an unknown (small, hundreds-scale) number of patients. **[verified counts; patient count undeterminable]**
- Models: DenseNet121 7.30M params; ResNet50 24.11M; EfficientNetB0 4.38M (notebook model summaries). **[verified]**
- Compute: experiment sweeps 10 epochs/config (250 steps/epoch); final runs ≤50 epochs with early stop; recorded final training times 79–111 min on 2×T4. **[verified]**
- Recorded experiment runs: 6 (DenseNet) + 9 (EfficientNet) + 12 (ResNet, two generations) + 3 final trainings. **[verified]**
- 1 repo; 7 notebooks; 3 results directories (~50 artifact files); 1 README. **0 tests, 0 CI workflows, 0 Docker files, 0 requirements files, 0 model weight files, 1 Git commit total.** **[verified]**

---

## Outcome

**What actually worked (verified):**
- The full transfer-learning experiment loop executed end-to-end and produced internally consistent artifacts: 3 architecture sweeps, selection algorithm, final trainings, test evaluations, per-class metrics, confusion matrices, prediction CSVs — all cross-checking cleanly (e.g., confusion matrix = 11 errors = test_predictions.csv = 0.99832).
- Block-wise freeze search behaved sensibly: freezing at block boundaries beat "freeze all" (0.8616 val) and informed LR/depth trade-offs.
- The DenseNet/ResNet final models are excellent **memorizers-and-near-neighbor-matchers** on this pool (99.7–99.8% on the leaked split).
- **The most valuable outcome:** the leakage was caught rather than shipped. The README's "Too Good To Be True?" section, the Final Report's "Do NOT deploy" recommendation, and the audit screenshot — and my independent re-analysis (84.1% of test images colliding with train by exact average-hash in a reproduced split) — all agree.

**Measurable results that exist:** test metrics above; V1→V2 delta (0.89 → 0.998) — verified numerically but **not a real improvement claim**, since both were measured on leaked data with different split protocols.

**Real-world/product impact:** none demonstrable. No deployment, no users, no clinical validation; the author explicitly recommends against deployment. **[verified absence]**

---

## Limitations

**What does NOT work:**
- The headline accuracy does not mean what it appears to mean: **test-set leakage is proven** (audit screenshot *and* my independent reproduction). Any claim of clinical-grade performance is void.
- EfficientNet inference path is broken/unexplained: val 0.99894 during training → 0.25455 after checkpoint reload, single-class predictions. Root cause never established; README's "distribution mismatch" explanation is **contradicted by the repo's own experiment data**.
- Nothing in the repo "runs" as shipped without manual recreation (see Current repository status).

**Unfinished / weaknesses:**
- *Modelling:* no subject-level splitting (patient identity unknown/untracked); single fixed seed; no cross-validation; no external test set; no error analysis beyond confusion matrix; no calibration/AUC; experiment phase used partial epochs (250 steps) with repeated data-exhaustion warnings; no class-weighting despite synthetic Moderate upsampling (which itself bakes a 64-original class into 10,000 images).
- *Methodology nuance:* selection-by-generalization-gap is unsound when val leaks; EfficientNet's two test failures (collapse + earlier `freeze_all` at 6.25% val — below random) went un-investigated.
- *Reproducibility:* no environment lock (TF 2.19.0 / Python 3.12 / Kaggle image only inferable from logs); `data_pipeline_setup.ipynb` is unexecuted; the actual train/val/test split consumed exists only as **private Kaggle datasets under three different slugs** (`dataset-finals`, `dataset-split`, `dementia-splits`); V1 pulled `archive.zip` from a private Google Drive; no dataset citation/license; no model weights committed.
- *Engineering debt:* experiment harness copy-pasted across 3 notebooks; `get_learning_rate_for_config()` hard-codes `total_layers=367`, which matches **none** of the three backbones (427/238/175) — LR tiers are computed on a wrong denominator (mostly harmless by luck); console prints mislabel total params as "Trainable params"; `trainable_layers=-35` nonsense value persisted in `densenet/results_final.csv` (freeze_all row); stale artifacts from the superseded ResNet experiment generation (`freeze_25/50/100/150_curves.png`) remain alongside current ones; README leads with "99.83%" and table claims DenseNet F1 "1.00" (computed value: 0.9984 — rounded claim overstated).
- *Documentation inconsistencies:* V1 markdown asserts class imbalance with Moderate as the smallest class, contradicted by its own executed EDA output (10,000/10,000/12,800/11,200); README's "EfficientNet — All Configs — 25.45%" row mislabels one collapsed final model as if every config failed (experiments reached 98.7% val); EDA declares "Dataset quality is excellent" before the later audit proved the opposite; EDA states dimensions "176×208", but my sample shows mixed sizes (majority 200×190) and mixed RGB/grayscale modes; V2 report claims "rigorous data splitting … to prevent data leakage" (Cell 3.1) one section before describing the leakage it failed to prevent.
- *Broken/dead integrations:* Colab Drive path, three private Kaggle dataset slugs, embedded GitHub user-attachment images in README (external hosting), no LICENSE file (repo or data).
- *Git hygiene:* entire project in one squashed commit with a misleading message; zero history.

---

## Current repository status

- **Runs locally:** **No, not as shipped.** Notebooks reference `/kaggle/input/...` (three distinct private Kaggle datasets) and `/content/drive/MyDrive/archive.zip`; TF ≠ pinned; GPU assumed. Because `dataset_images/` *is* included, a motivated user could re-run `split_dataset()` logic and retrain, but nothing is packaged to make that work out of the box, and the split notebook itself was never executed in-repo. **[verified]**
- **Tests pass:** **N/A — no tests exist.** **[verified]**
- **Deployment:** none exists or is claimed to exist; repo explicitly says do not deploy. **[verified]**
- **README quality:** **Good narrative quality, unusually honest about the leakage failure — but still leads with the leaked 99.83% headline, contains the overstated F1 "1.00", the misleading EfficientNet "All Configs 25.45%" row, and an unproven root-cause story.** Mixed accuracy: strong self-awareness, some unsupported causal claims. **[verified by cross-checking every quantitative claim in it against artifacts]**
- **Demo available:** none (no app, no video, no live endpoint). "Demo" = static PNGs/CSVs and README figures. **[verified absence]**

---

## Evidence

**Relevant files (all paths relative to repo root):**
- `README.md` — claims audited line-by-line against artifacts.
- `notebooks/Alzheimer's_densenet-experiments-and-final-training.ipynb` — executed (exec 1–17): sweep, selection, final training, test eval (test loss 0.0050, acc 0.9983).
- `notebooks/Alzheimer's_resnet-experiments-and-final-training.ipynb` — executed (exec 1–19): two experiment generations (cells 26 and 28–29), final test acc 0.9971.
- `notebooks/Alzheimer's_efficientnet-experiments-and-final-training.ipynb` — executed (exec 1–17): sweep incl. 0.98687 val config; final run val 0.99894; collapsed test 0.2545 with single-class classification report.
- `notebooks/Alzheimer's_data_pipeline_setup.ipynb` — splitting code, **unexecuted**.
- `notebooks/Alzheimer's_Disease_Classification_EDA_Improved.ipynb` — executed EDA.
- `notebooks/Alzheimer's_Disease_Classification_Initial_Project_Full.ipynb` — V1; preserved logs incl. 0.8894–0.8906 val acc and classification report (0.89, n=8,800).
- `notebooks/Alzheimer's_Disease_Final_Project_Report.ipynb` — markdown-only report; audit screenshot attachment (Cell 7).
- `results_and_visuals/{densenet,resnet50,efficientnet}/` — `results_final.csv`, `per_class_metrics.csv`, `test_predictions.csv` (6,600 rows, 6,601 lines), `*_config.json` (timestamps 2026-02-01), curve PNGs, confusion matrices. Cross-verified: identical test filenames/order across all three models (fair same-leak comparison).
- `dataset_images/` — 44,000 images; duplication statistics measured directly (below).
- Git: single commit `97bec18e`; GitHub: created 2026-01-23, pushed 2026-05-26, 0 PRs/issues/stars/forks, no license.

**Independent verification performed for this document (not part of the repo):**
Using only the shipped `dataset_images/` with Pillow/ImageHash/sklearn in a scratch venv: exact-byte duplicates 0.1–3.5% (3,000-image samples/class); 6,000-image stratified sample → 1,157 unique average-hashes (87.1% in duplicate groups); repo's exact split procedure (stratified 70/15/15, seed 42) reproduced on that sample → **84.1% of test images have an identical-hash counterpart in train** (author's screenshot said 683/1000 = 68.3% on an alphabetically biased sample — same conclusion, larger effect). Also verified: mixed image sizes/modes; filename-family census; every numeric claim in README against executed outputs/CSVs.

**PRs:** none (0 open/closed — checked via `gh pr list --state all`). **Issues:** none. **Screenshots:** only the audit screenshot embedded in the Final Report notebook; no app screenshots exist (no app). **Demo:** none. **Presentation:** none in repo (V1 markdown contains presentation-style summaries; Final Report notebook is the closest to a written report).

---

## Candidate CV bullets

*(Intentionally left blank.)*

---

## Work required

### Critical
*(Problems that prevent the repository from being credible professional evidence.)*
1. **Reframe or contextualize the 99.83% headline** — it is currently the first number a reviewer sees; without the caveat it invites the wrong conclusion before the audit section rescues it.
2. **Correct the EfficientNet narrative in README** — "All Configs | 25.45%" is factually wrong for the sweep (98.7% val); either investigate the checkpoint-reload collapse properly ( BN-freezing test, val-after-reload test) or state the failure honestly as *unexplained train/serve skew*.
3. **Convert the audit from a screenshot into committed, executable code** (a small `audit_duplicates.py` or notebook cell with preserved outputs and numbers) — currently the single most important finding rests on an image.
4. **Document dataset provenance** — source URLs, original train/test counts, and a license/usage note; without this the repo ships 482 MB of medical images of unknown licensing.
5. **Make the pipeline reproducible**: commit the executed split (or the exact `train/validation/test` file lists as CSV), pin the environment (`requirements.txt` with TF 2.19.0), and replace the three private Kaggle dataset slugs + Colab Drive path with repo-relative paths.

### High value
6. Add a tiny automated check that can fail (e.g., script asserting `test_predictions.csv` accuracy equals reported metrics; a leakage test on any new dataset) — gives the repo *some* executable verification instead of zero tests.
7. Extract the duplicated experiment harness into one module used by all three architecture notebooks; fix `get_learning_rate_for_config(total_layers=367)` and the mislabeled "Trainable params" print.
8. Publish the final DenseNet `.h5` (or `.keras`) weights + a single-image inference example cell — proves the artifact exists and works.
9. Clean stale ResNet artifacts (`freeze_25/50/100/150_curves.png`) or document the two experiment generations.
10. Fix metric presentation: report macro-F1 = 0.9984 (not "1.00"); add per-run seeds/variance note.

### Optional
11. Add a LICENSE (code) and DATA_LICENSE/attribution note.
12. Squash-or-annotate the git story: a short `CHANGELOG.md` or tag `v1-baseline`/`v2-audit` since real history is unrecoverable.
13. Fix the Unicode apostrophe filename (`Alzheimer’s_...`) for cross-platform tooling.
14. Add a one-paragraph "How to verify this repo yourself" section (hash-audit instructions) — turns the strongest story (integrity auditing) into a runnable demonstration.

---

## Professional Evidence Assessment

1. **What engineering capability does this project prove best?**
   Building a disciplined deep-learning experiment workflow solo: systematic architecture/config sweeps with a reusable harness, callback-driven training, selection-by-criterion (not best-last-epoch), full artifact persistence, and end-to-end execution on cloud GPUs with internally consistent results. Equally, it demonstrates **measurement skepticism** — catching and correctly diagnosing evaluation-data leakage post-hoc.
2. **What ML/AI capability does it prove best?**
   Practical transfer learning on medical images: per-image robust (median/IQR) intensity normalization, block-wise fine-tuning depth search across three backbones, LR scheduling by trainable ratio, and honest per-class metric reporting. The data-integrity audit (duplicate/near-duplicate detection with perceptual hashing) is the most differentiated ML-engineering skill demonstrated.
3. **What is the strongest verifiable achievement?**
   The **correctly identified and now independently confirmed dataset leakage diagnosis** ("99.83% was too good to be true — the test set was contaminated by pre-augmented duplicates; do not deploy"). It is the only claim in the repo that survives full independent reproduction, and it demonstrates judgment that most portfolio projects lack.
4. **What is the biggest credibility weakness?**
   The headline 99.83%/99.7% numbers are products of the leaked split and could be quoted out of context; combined with zero tests/CI/environment pinning, an unexecuted pipeline notebook, private-data dependencies, screenshot-only audit code, and the internally inconsistent EfficientNet story, a careful reviewer will discount much of the performance narrative. (The single squashed commit also means no process evidence.)
5. **Is there enough here to justify investing additional time in this project?**
   Yes — but narrowly. The salvageable asset is *methodology + integrity narrative*, not the model. A small, bounded investment (items 1–5 under Critical: honest README framing, executable audit, provenance+reproducibility fixes) converts this from "Kaggle notebook project with suspicious results" into "evidence of rigorous experimental practice and data forensics". Building new features would not add comparable evidence value; retraining on a clean non-augmented dataset would, but that is a new sub-project.
6. **What type of professional role would this project support as evidence?**
   Junior-to-mid **Applied ML Engineer / ML Engineer (computer vision)** roles, and **data-centric AI / ML quality** positions — as one supporting project, not a flagship. It supports claims about experiment design, transfer learning, and dataset validation. It does **not** support claims about production ML systems, MLOps tooling, deployment, or collaborative engineering (no tests, CI, serving, history, or teammates).
7. **What claims should I NOT make publicly based on the current repository?**
   - "Built a 99.8%-accurate Alzheimer's classifier" (leaked test set; even the repo disowns it).
   - Any clinical/screening readiness, patient-level generalization, or deployment impact.
   - "Solved covariate shift / proved EfficientNet unsuitable" — root cause unproven and contradicted by own data.
   - "Achieved macro-F1 1.00" (actual: 0.9984 on leaked data).
   - "Improved the model from 89% to 99.8%" as a genuine capability gain (different protocols, both leaked).
   - "Rigorous MLOps practices" in the tooling sense — there are no tests, CI, containers, env locks, or model registry; the rigor shown is analytical, not infrastructural.
   - Dataset ownership/"curated a 44k-image medical dataset" — third-party data of undocumented provenance, effectively ~6.4k originals bulk-augmented.
