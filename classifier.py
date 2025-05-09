# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings 
warnings.filterwarnings("ignore")

# Loading the Dataset
data=pd.read_csv("iris.csv")
print("\n",data.head())
print("\n",data.describe())

# Features & Target
features=data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
target=data["Species"]

# Splitting the Dataset
x_test,x_train,y_test,y_train=train_test_split(features,target,test_size=0.2,random_state=42)
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

# Classifier Prediction on Test Data
y_pred=classifier.predict(x_test)
print("\nAccuracy on test data:",accuracy_score(y_test,y_pred))
print("\nClassification Report\n:",classification_report(y_test,y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",xticklabels=classifier.classes_, yticklabels=classifier.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

X = data[["PetalLengthCm", "PetalWidthCm"]].values
y = LabelEncoder().fit_transform(data["Species"])

model = LogisticRegression()
model.fit(X, y)

# Creating a Mesh

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Decision Boundaries (Logistic Regression)')
plt.show()

# Sample Prediction
new_pred=classifier.predict([[5.4,2.6,4.1,1.3]]) [0]
print("Predicted Species:",new_pred)
