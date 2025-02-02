# Card-fraud-detection-project
R&amp;D of LLMs for fraud detection task

This repository provides multiple fraud detection models utilizing pre-trained BERT and sequence models (GRU, LSTM) for card transaction data. It includes scripts for data preprocessing, model training, and evaluation. The repository also contains the essential code for integrating various models with advanced techniques such as attention mechanisms and sequence-based architectures. 
****
##  Repository Structure:

- **data/**
  - `ClassificationDataCollator1.py`: Data collator script for handling classification-specific batching and padding.
  - `ClassificationDataCollator_v2.py`: An updated version of the classification data collator.
  - `TransactionDataCollator.py`: A data collator preparing for bert training.
  - `TransactionDataset.py`: Dataset script for handling transaction-specific data, including reading and processing data.
  - `helpers.py`: Utility functions for preprocessing

- **models/**
  - Contains different model architectures (e.g., BERT + GRU, BERT + LSTM with Attention) for fraud detection tasks, including training and evaluation logic.


- **vocabulary/**

- **Big data project.pdf**
  - A PDF presentation describing the overall project, including methodology, experiments, and results.

- **big-data-continue-2.ipynb**
  - A Jupyter notebook with project.


