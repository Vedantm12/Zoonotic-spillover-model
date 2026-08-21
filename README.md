# Zoonotic-spillover-model
A generalized machine learning pipeline using scikit-learn to rapidly train and evaluate Logistic Regression models on epidemiological and zoonotic disease datasets.
# Generalized Disease ML Classifier

This repository contains a generalized machine learning pipeline designed to quickly train Logistic Regression models on epidemiological datasets (such as Nipah virus data). Built to streamline exploratory data analysis in computational biology research, the script allows users to easily swap datasets and target variables via a command-line interface without rewriting any code.

## Features
* **Interactive CLI**: Prompts the user to input the target CSV file, target prediction column, and specific features directly in the terminal.
* **Automatic Feature Selection**: If no specific features are provided, the script automatically drops the target column and filters the remaining data to strictly include numerical and boolean values, preventing pipeline crashes from text columns.
* **Robust Preprocessing**: Automatically handles missing data (`dropna()`) and scales numerical features (`StandardScaler`) before training.
* **Class Imbalance Handling**: Uses `class_weight='balanced'` within the Logistic Regression model to accurately evaluate datasets where positive disease cases might be rare.
* **Automated Reporting**: Generates a classification report and accuracy score, printing it to the terminal and automatically saving a timestamped copy to a `.txt` file for easy record-keeping.

## Prerequisites
Ensure you have Python installed along with the following libraries:
```bash
pip install pandas scikit-learn
