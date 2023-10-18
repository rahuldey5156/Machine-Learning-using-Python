import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#sns gives extensive colourful plot facilities and also gives many statistical functions

path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

#gives column names
headernames = ['sepal-length','sepal-width','petal-length','petal-width','Class']

data = pd.read_csv(path, names = headernames)
print(data.shape)

#print(data.head())

#data preprocessing
X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values

#Dividing data into train and test splits, with 70% of training data and 30% of testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

#data scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#training the model with DecisionTreeClassifier class of sklearn
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

#Make the prediction
y_pred = classifier.predict(X_test)

#print the result
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)

sns.heatmap(result,
            annot = True,
            fmt = 'g',
            xticklabels = ['Setosa','Versicolor','Virginica'],
            yticklabels = ['Setosa','Versicolor','Virginica'])

plt.ylabel('Prediction', fontsize = 13)
plt.xlabel('Actual', fontsize = 17)
plt.title('Confusion Matrix', fontsize = 17)
plt.show()

result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print(result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)



