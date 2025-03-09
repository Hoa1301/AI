# train_model.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pickle

# Đọc dataset từ file CSV

dataset = pd.read_csv('cityu10c_train_dataset.csv')

# Xử lý missing values
dataset = dataset.dropna(axis=1, thresh=int(0.8 * len(dataset)))
dataset.fillna(dataset.select_dtypes(include=['number']).median(), inplace=True)

# Xử lý dữ liệu categorical
categorical_cols = dataset.select_dtypes(include=['object']).columns
for col in categorical_cols:
    dataset[col] = dataset[col].fillna('Unknown')

# Chia tập dữ liệu
features = ['Age', 'AnnualIncome', 'CreditScore', 'EmploymentStatus', 'EducationLevel', 'LoanAmount', 'LoanDuration']
target = ['LoanApproved']
X = dataset[features]
y = dataset[target]

# Định nghĩa các transformer
categorical_features = ['EmploymentStatus', 'EducationLevel']
numerical_features = ['Age', 'AnnualIncome', 'CreditScore', 'LoanAmount', 'LoanDuration']

numerical_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Tạo pipeline và huấn luyện mô hình
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

pipeline.fit(X, y.values.ravel())


pickle.dump(pipeline, open("decision_tree_pipeline.pkl", 'wb'))