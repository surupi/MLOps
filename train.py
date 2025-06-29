import os
import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('iris.csv')

train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = train.species
X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_test = test.species

print('Training start')
mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
mod_dt.fit(X_train, y_train)
prediction = mod_dt.predict(X_test)
print('Training Complete')

accuracy = metrics.accuracy_score(prediction, y_test)
print('The accuracy of the Decision Tree is', "{:.3f}".format(accuracy))

os.makedirs('models', exist_ok=True)
with open('models/week2_decisionTree_model.pkl', 'wb') as f:
    pickle.dump(mod_dt, f)
print('Model saved')

with open('metrics.txt', 'w') as f:
    f.write(f"Test Accuracy: {accuracy:.3f}\n")
print('Metrics saved')
