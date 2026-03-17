# 11685-Guided-Project README

## Project Goal
This repository is for the 11-685 Guided Project. We currently focus on two main tasks:
1. EEG classification: predict image category from EEG signals
2. EEG retrieval: map EEG features into a shared space for caption / image retrieval


### Main principles:
- notebooks are only for exploration and plotting
- reusable code goes into src/
- runnable entry scripts go into scripts/
- experiment settings go into configs/
- outputs go into outputs/
- raw large datasets should not be uploaded to GitHub



## 1. Recommended Folder Structure


11685-Guided-Project/
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── classification_baseline.yaml
│   ├── classification_cnn.yaml
│   └── retrieval_baseline.yaml
│
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── notebooks/
│   ├── 00_data_check.ipynb
│   ├── 01_baseline_debug.ipynb
│   └── 02_report_figures.ipynb
│
├── scripts/
│   ├── build_metadata.py
│   ├── make_splits.py
│   ├── train_classification.py
│   ├── eval_classification.py
│   ├── train_retrieval.py
│   └── eval_retrieval.py
│
├── src/
│   ├── data_utils.py
│   ├── datasets.py
│   ├── models.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train_utils.py
│   └── retrieval_utils.py
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── figures/
│   └── predictions/
│
└── reports/
    ├── midterm/
    └── final/


## 2. What Each Folder Is For

1. configs/
Purpose:
Stores experiment configuration files, such as model type, batch size, learning rate, number of epochs, and data paths.

Why this matters:
Teammates can change experiment settings without editing the training code directly.

Typical files:
- classification_baseline.yaml
  config for the classification baseline
- classification_cnn.yaml
  config for a CNN-based classification model
- retrieval_baseline.yaml
  config for a retrieval experiment


2. data/
Purpose:
Stores data-related content, but raw large files should not be committed to GitHub.

Subfolders:
- raw/
  location of raw data; usually only the folder structure is kept in the repo
- processed/
  processed files such as metadata.csv, cleaned index tables, or cached outputs
- splits/
  train / val / test split files, such as csv or json

Suggestion:
Add a small data/README.txt to explain where the real data should be placed.


3. notebooks/
Purpose:
Used for exploration, debugging, and generating figures for the report.

Good use cases:
- checking EEG tensor shapes
- checking class distribution
- plotting learning curves
- plotting confusion matrices
- generating report figures

Not good for:
- final training pipeline
- large repeated code blocks
- hidden project logic that only one teammate understands

Rule:
If code will be reused later, move it into src/.


4. scripts/
Purpose:
Stores the runnable entry-point scripts.

You can think of this folder as:
the place where teammates directly run commands.

Typical scripts:
- build_metadata.py
  build a unified metadata table from raw EEG, labels, and captions
- make_splits.py
  generate train / val / test splits
- train_classification.py
  train the classification model
- eval_classification.py
  evaluate classification results
- train_retrieval.py
  train the retrieval model
- eval_retrieval.py
  evaluate retrieval results

Why this matters:
A new teammate does not need to understand the whole codebase before running experiments.


5. src/
Purpose:
Stores the main reusable project code.

You can think of this folder as:
the real code library of the project.

Recommended files:
- data_utils.py
  helper functions for preprocessing, loading metadata, and handling paths
- datasets.py
  PyTorch Dataset classes for reading samples
- models.py
  model definitions such as baseline MLP, CNN, and projection heads
- losses.py
  loss functions such as cross entropy and contrastive loss
- metrics.py
  evaluation metrics such as accuracy, Recall@K, and confusion matrix
- train_utils.py
  shared training utilities such as train_one_epoch, validate, and checkpoint saving
- retrieval_utils.py
  retrieval-specific functions such as embedding extraction, similarity computation, and top-k search

Rule:
Code here should be reusable by both scripts/ and notebooks/.


6. outputs/
Purpose:
Stores all experiment outputs.

Subfolders:
- checkpoints/
  saved model weights
- logs/
  training logs
- figures/
  plots such as loss curves and confusion matrices
- predictions/
  prediction files, submissions, retrieval results

Suggestion:
Large output files usually should not be committed to GitHub.


7. reports/
Purpose:
Stores report-related materials.

Suggested usage:
- midterm/
  draft figures, tables, and notes for the midterm report
- final/
  materials for the final report


## 3. What the Key Files Are For


1. requirements.txt
Lists the required Python packages.
A new teammate can install the environment from this file.

2. .gitignore
Tells Git which files should not be uploaded, such as:
- raw datasets
- large checkpoints
- temporary cache files
- notebook-generated artifacts

3. README_CN.txt / README_EN.txt
Project documentation.
A new teammate should read one of these first.


## 4. Recommended Data Flow

The project should follow this pipeline:

raw data
-> build_metadata.py
-> metadata / processed index
-> make_splits.py
-> train / val / test split files
-> datasets.py + DataLoader
-> model
-> loss + metrics
-> outputs/
-> reports/

Expanded explanation:

Step 1: Put raw data into data/raw/
This may include:
- EEG numpy files
- image ID mapping files
- class labels
- caption annotations

Step 2: Run scripts/build_metadata.py
Purpose:
merge scattered raw information into one unified table, such as metadata.csv.
Each row may contain:
- subject_id
- session_id
- trial_id
- eeg_path
- class_label
- caption
- image_id

Step 3: Run scripts/make_splits.py
Purpose:
generate train, validation, and test splits from metadata.csv.
Outputs are saved into data/splits/.

Step 4: Training scripts read split files
train_classification.py or train_retrieval.py will:
- load metadata
- load split files
- build Dataset and DataLoader
- train the model

Step 5: Evaluation scripts save results
eval_classification.py or eval_retrieval.py will generate:
- accuracy / recall
- confusion matrix
- retrieval results
- figures and prediction files

Step 6: Move figures and tables from outputs/ into reports/
These materials are then used in the midterm or final report.


## 5. Suggested Onboarding Order for New Teammates

Recommended order:

1. Read README_CN.txt or README_EN.txt
2. Read data/README.txt and confirm data paths
3. Run build_metadata.py
4. Run make_splits.py
5. Run train_classification.py
6. Then move on to the retrieval code

This keeps the codebase much less overwhelming.


## 6. Minimal Collaboration Rules

1. Do not upload raw data
2. Do not use notebooks as the main codebase
3. Keep training logic inside scripts/ and src/
4. Keep each code change focused on one purpose
5. Prefer adding a new config file instead of overwriting someone else's config
6. Save results under outputs/ so report writing is easier later


## 7. Suggested Minimum Working Workflow

Step 1:
Prepare data/raw/

Step 2:
Run
python scripts/build_metadata.py

Step 3:
Run
python scripts/make_splits.py

Step 4:
Run
python scripts/train_classification.py --config configs/classification_baseline.yaml

Step 5:
Run
python scripts/eval_classification.py --config configs/classification_baseline.yaml

Step 6:
If retrieval is needed, run
python scripts/train_retrieval.py --config configs/retrieval_baseline.yaml


## 8. Summary


The core design is simple:

- data/ stores data
- configs/ stores settings
- src/ stores reusable code
- scripts/ stores runnable entry points
- outputs/ stores experiment results
- reports/ stores report materials
