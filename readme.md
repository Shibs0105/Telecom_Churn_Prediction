Telecom Churn Prediction

Project Overview

In the highly competitive telecom industry, customer churn is a major challenge as it significantly impacts revenue. Customer retention is crucial, especially for high-value customers, as acquiring new customers is much costlier. This project focuses on predicting churn among high-value prepaid customers in the Indian and Southeast Asian markets using customer usage data from four months.

Business Problem

Telecom companies face high churn rates, with prepaid customers often switching to competitors without notice. The goal of this project is to predict churn for high-value customers, enabling proactive retention strategies and reducing revenue loss.

Dataset

The dataset contains customer-level information for four consecutive months (June, July, August, and September), encoded as 6, 7, 8, and 9 respectively. We aim to predict churn in the last month (September) using data from the previous three months.

Key Steps

Data Cleaning and Preprocessing: Handling missing values, removing irrelevant columns, and identifying high-value customers based on recharge amounts.

Churn Tagging: Marking churned customers based on zero usage of calls and internet in the churn month.

Modeling:

Logistic Regression

Random Forest Classifier

Hyperparameter Tuning and Model Evaluation

Performance Evaluation:

Classification Report

ROC-AUC Curve

Results

The models achieved satisfactory performance in predicting churn with the Random Forest Classifier showing a higher ROC-AUC score. The ROC curve provides a visual representation of the model's accuracy.

Tools and Libraries

Python (pandas, numpy, sklearn, imblearn)

Data Visualization (matplotlib, seaborn)

Classification Models (Logistic Regression, Random Forest)

How to Run

Clone the repository.

Install necessary libraries using pip install -r requirements.txt.

Run the script using:

python telecom_churn.py

Review the output and ROC curve for model performance.

Future Enhancements

Implement more advanced machine learning models.

Fine-tune hyperparameters for better accuracy.

Incorporate customer demographics and location data for deeper insights.

License

This project is licensed under the MIT License.

