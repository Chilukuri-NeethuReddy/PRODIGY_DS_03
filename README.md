# PRODIGY_DS_03

## ProdigyInfoTech_TASK3

## TASK 3: Decision Tree Classifier for Customer Purchase Prediction
This project demonstrates how to build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data from the Bank Marketing dataset.

## Dataset:
The dataset used in this project is the Bank Marketing dataset, available on the UCI Machine Learning Repository.

## Steps:

1.**Data Loading**: Load the dataset and inspect columns.

2.**Data Preprocessing**: Encode categorical variables and split data into training and testing sets.

3.**Model Building**: Build a decision tree classifier with specified pruning parameters.

4.**Model Evaluation**: Evaluate the classifier using accuracy score and classification report.

5.**Visualization**: Visualize the decision tree to understand its structure.

## How to Run:
1.Clone the repository.

2.Ensure you have the necessary libraries installed (pandas, scikit-learn, matplotlib).

3.Place the dataset (bank.csv) in the same directory as the script.

4.Run the script (decision_tree_classifier.py) to train the model and generate visualizations.

## Code

```python

# Import necessary libraries

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pyplot as plt

# Load the dataset with the correct delimiter

df = pd.read_csv('bank.csv', sep=';')

# Print columns to inspect their names and contents

print("Columns in the dataset:")

print(df.columns)
```

![Screenshot 2024-07-07 104727](https://github.com/Chilukuri-NeethuReddy/PRODIGY_DS_03/assets/174725064/5deabde5-aae6-4aae-9a1a-9fafd05d2e33)

```python

# Assuming 'y' is the target variable indicating purchase (yes/no)

X = df.drop(columns=['y'])  # Features

y = df['y']                 # Target variable

# Encode categorical variables

categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

X_encoded = pd.get_dummies(X, columns=categorical_cols)

print(X_encoded)
```

![Screenshot 2024-07-07 104850](https://github.com/Chilukuri-NeethuReddy/PRODIGY_DS_03/assets/174725064/3551cbcc-8231-4bfb-a7eb-529a3aa536a5)

![Screenshot 2024-07-07 104907](https://github.com/Chilukuri-NeethuReddy/PRODIGY_DS_03/assets/174725064/b005e490-e91b-46d7-974d-31326e09f9c2)

```python

# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize decision tree classifier with pruning parameters

clf = DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_split=20, min_samples_leaf=10)

# Train the classifier

clf.fit(X_train, y_train)

# Predict on the test data

y_pred = clf.predict(X_test)

# Evaluate the model

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")

print(classification_report(y_test, y_pred))
```

![Screenshot 2024-07-07 105030](https://github.com/Chilukuri-NeethuReddy/PRODIGY_DS_03/assets/174725064/2ad291d6-3dce-46ce-9a02-e72b98bceeb5)

```python

# Visualize the decision tree

plt.figure(figsize=(20, 10))  # Set the figure size

plot_tree(clf, filled=True, feature_names=list(X_encoded.columns), rounded=True, fontsize=12, class_names=['No Purchase', 'Purchase'])

plt.show(block=True)
```

###   Decision tree
![Screenshot 2024-07-07 105145](https://github.com/Chilukuri-NeethuReddy/PRODIGY_DS_03/assets/174725064/406faab9-1081-45a9-872a-675698071301)






