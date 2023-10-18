import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

data = pd.read_csv(path, names = headernames)
print(data.shape)

print(data.head())

#Accessing index
X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values

#Here we are dividing data into train and test splits, with 70% of training data and 30% o testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

#Data Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#training the model with the help of GaussianNB class of sklearn
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train);

#make prediction
y_pred = classifier.predict(X_test)

#print results
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:")
print(result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:", result2)



