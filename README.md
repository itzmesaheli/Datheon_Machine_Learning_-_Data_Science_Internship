# Datheon Machine Learning & Data Science Internship

This repository contains my completed tasks for the Datheon Machine Learning & Data Science Internship.

## Repository Structure

```
Datheon_Machine_Learning_Internship/
├── README.md
├── Task1_Iris_Classification/
│   └── Task1_Iris_Classification_Pipeline.ipynb
├── Task2_Housing_Price_Prediction/
│   └── Task2_Housing_Price_Regression.ipynb
└── Task3_Customer_Churn_Classifier/
    └── Task3_Customer_Churn_Classifier.ipynb
```

## Tasks Completed

### Task 1: Multi-Class Benchmark Classification Pipeline (Iris)
- Dataset: Iris (loaded from Scikit-Learn)
- Steps: EDA with Seaborn pairplots, feature scaling (StandardScaler), stratified 80/20 train/test split
- Models: Logistic Regression, Decision Tree
- Metrics: Accuracy, Precision, Recall, F1-score
- Visualizations: Confusion matrix heatmaps, decision boundary plots, decision tree diagram
- Folder: `Task1_Iris_Classification/`

### Task 2: Real Estate Housing Price Prediction (Ridge & Lasso Regression)
- Dataset: California Housing (loaded from Scikit-Learn)
- Steps: EDA, outlier capping, feature engineering (bedroom ratio, rooms per person, log population, income squared, income-age interaction), feature scaling
- Models: Linear Regression, Ridge, Lasso (regularization strength chosen with 5-fold cross-validation)
- Metrics: R2, MAE, RMSE
- Analysis: Coefficient comparison, Lasso feature selection, residual error analysis (residual plots, histogram, Q-Q plot)
- Folder: `Task2_Housing_Price_Prediction/`

### Task 3: Predictive Customer Churn Classifier (Random Forest & XGBoost)
- Problem: Predict customer churn on a heavily imbalanced dataset (about 10% churners)
- Imbalance handling: Stratified split, Random Forest `class_weight`, XGBoost `scale_pos_weight`
- Models: Random Forest (baseline and class-weighted), XGBoost (default and tuned)
- Tuning: `RandomizedSearchCV` optimizing ROC-AUC, plus decision threshold tuning on out-of-fold training predictions
- Metrics: ROC-AUC, PR-AUC, Precision, Recall, F1-score
- Visualizations: Class distribution, ROC curves, confusion matrix, feature importances
- Folder: `Task3_Customer_Churn_Classifier/`

## Tools Used
Python, Pandas, NumPy, Scikit-Learn, XGBoost, Seaborn, Matplotlib, SciPy, Jupyter Notebook

## How to Run
1. Clone the repository
2. Install dependencies: `pip install pandas numpy scikit-learn xgboost seaborn matplotlib scipy jupyter`
3. Open any notebook in Jupyter or Google Colab and click **Run All**

## Author
Saheli Sarkar
