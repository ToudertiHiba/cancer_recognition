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
1. Install Poetry for dependency management: [Poetry Installation Guide](https://python-poetry.org/docs/#installation).
2. Run `poetry install` to install the project dependencies.
3. Ensure you have the necessary dataset (`cancer.csv`) in the project directory.
``


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
