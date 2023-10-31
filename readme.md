# Tumor Classification Project

## Overview

This project focuses on developing a machine learning model for classifying tumors as either cancerous or non-cancerous based on a dataset from Kaggle. The model is built using TensorFlow and the data preprocessing is handled with pandas.

## Project Structure

- `cancer_recognition.ipynb`: Jupyter notebook containing code and explanations for the tumor classification project.
- `cancer_recognition.py`: Python script containing the main code for the tumor classification project.
- `cancer.csv`: The dataset used for training and testing the model.
- `pyproject.toml`: Poetry configuration file containing dependencies and project metadata.
- `README.md`: This file providing an overview of the project and instructions for running it.

## Getting Started

To get started with the project, follow these steps:

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/yourusername/tumor-classification.git
   cd tumor-classification
   ```

2. **Install Dependencies**:
   ```bash
   poetry install
   ```

3. **Download Dataset**:
   - Download the dataset `cancer.csv` from [Kaggle](https://www.kaggle.com/datasets) and place it in the project directory.

4. **Run the Main Script**:
   ```bash
   python cancer_recognition.py
   ```

## Dependencies

This project uses the following libraries:

- pandas
- TensorFlow
- scikit-learn

All dependencies are managed via Poetry. You can find the complete list of dependencies in the pyproject.toml file.

## Notes

- The model architecture consists of an input layer, two hidden layers, and an output layer, all utilizing the sigmoid activation function.
- The model is compiled using binary cross-entropy loss and the Adam optimizer, suitable for binary classification tasks.
- Evaluation metrics including accuracy, precision, recall, F1-score, and the confusion matrix are calculated.
