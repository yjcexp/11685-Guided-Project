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

```
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
```

- data/ stores data
- configs/ stores settings
- src/ stores reusable code
- scripts/ stores runnable entry points
- outputs/ stores experiment results
- reports/ stores report materials
