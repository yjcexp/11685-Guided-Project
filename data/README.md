Data Folder Structure
=====================

This folder contains all data-related files for the EEG classification and retrieval project.

Folder Structure:
-----------------
- raw/          : Raw data files (EEG numpy files, labels, captions, etc.)
                 NOTE: This folder should not be committed to GitHub

- processed/    : Processed data files such as metadata.csv, cleaned index tables

- splits/       : Train/val/test split files (CSV or JSON format)

Instructions:
-------------
1. Place raw data in data/raw/
2. Run python scripts/build_metadata.py to generate processed/metadata.csv
3. Run python scripts/make_splits.py to generate split files in data/splits/
