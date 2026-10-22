# Task 1: Multi-Class Benchmark Classification Pipeline (Iris)

## Overview
This project builds an end-to-end supervised classification pipeline on the Iris dataset. It covers exploratory analysis, feature scaling, model training, and benchmarking of two classifiers: Logistic Regression and a Decision Tree.

## Problem Statement
Classify iris flowers into three species (Setosa, Versicolor, Virginica) from four measurements: sepal length, sepal width, petal length and petal width.

## Dataset
- **Iris dataset**, loaded directly from Scikit-Learn (`load_iris`)
- 150 samples, 4 numeric features, 3 balanced classes
- No missing values

## Approach
1. **Load data** with Pandas
2. **Exploratory data analysis** using Seaborn pairplots to inspect class separability
3. **Train/test split**: 80/20, stratified so each class keeps the same proportion
4. **Feature scaling** with `StandardScaler`, fit on the training set only to avoid data leakage
5. **Model training**: Logistic Regression and a Decision Tree (`max_depth=3`)
6. **Evaluation**: accuracy, macro precision, recall and F1-score, plus a full classification report
7. **Visualization**: confusion matrix heatmaps, decision boundary plots (using petal length and petal width), and the decision tree diagram

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~93.3% |
| Decision Tree | ~96.7% |

- Setosa is classified perfectly by both models.
- Almost all errors occur between Versicolor and Virginica, which overlap in feature space.
- Exact metrics (precision, recall, F1) are shown in the notebook output.

## Files
| File | Description |
|---|---|
| `Task1_Iris_Classification_Pipeline.ipynb` | Complete notebook with code, metrics and plots |

## Tools Used
Python, Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib, Jupyter Notebook

## How to Run
1. Install the dependencies:
```
   pip install pandas numpy scikit-learn seaborn matplotlib jupyter
```
2. Open `Task1_Iris_Classification_Pipeline.ipynb` in Jupyter or Google Colab
3. Click **Run All**

All results use `random_state=42`, so they are reproducible.

## Key Takeaways
- Scaling features and fitting the scaler on training data only is important to avoid data leakage.
- Logistic Regression gives a smooth linear decision boundary, while the Decision Tree gives easy-to-read axis-aligned splits but can overfit without a depth limit.
- Accuracy alone can hide per-class mistakes, so precision, recall, F1 and the confusion matrix give a fuller picture.

## Author
Saheli Sarkar
