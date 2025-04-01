# Telecom Churn Prediction - Main Python Script

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Step 1: Load the Data
file_path = 'telecom_churn_data.csv'
data = pd.read_csv(file_path)

# Step 2: Data Cleaning and Preprocessing
# Drop columns with more than 30% missing values
data = data.dropna(thresh=len(data) * 0.7, axis=1)

# Drop mobile number column (not relevant)
data = data.drop(columns=['mobile_number'], errors='ignore')

# Replace missing values with zero or mean
data.fillna(0, inplace=True)

# Step 3: Identify High-Value Customers
recharge_cols = ['total_rech_amt_6', 'total_rech_amt_7']
data['avg_rech_amt_6_7'] = data[recharge_cols].mean(axis=1)
high_value_threshold = data['avg_rech_amt_6_7'].quantile(0.70)
high_value_customers = data[data['avg_rech_amt_6_7'] >= high_value_threshold]

# Step 4: Tag Churners
churn_conditions = (
    (high_value_customers['total_ic_mou_9'] == 0) &
    (high_value_customers['total_og_mou_9'] == 0) &
    (high_value_customers['vol_2g_mb_9'] == 0) &
    (high_value_customers['vol_3g_mb_9'] == 0)
)
high_value_customers['churn'] = churn_conditions.astype(int)

# Drop churn month columns
churn_month_cols = [col for col in high_value_customers.columns if '_9' in col]
high_value_customers = high_value_customers.drop(columns=churn_month_cols)

# Step 5: Train-Test Split
X = high_value_customers.drop(columns=['churn'])
y = high_value_customers['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Model Building - Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Step 7: Evaluation
print('Logistic Regression Report:')
print(classification_report(y_test, y_pred_lr))

# ROC-AUC
roc_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])
print(f'ROC-AUC: {roc_auc:.2f}')

# Step 8: Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print('Random Forest Report:')
print(classification_report(y_test, y_pred_rf))

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Random Forest (AUC = {:.2f})'.format(roc_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

print('Telecom Churn Prediction Script Completed Successfully')
