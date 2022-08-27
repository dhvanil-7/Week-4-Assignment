## Importing required modules
from sklearn.datasets import load_iris
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Loading data into the dataframe dataset
dataset = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
target_mapping={0:'setosa', 1:'versicolor', 2:'virginica'}
dataset['Target'] = load_iris()['target']
dataset['Target_name'] = dataset['Target'].map(target_mapping)

# Defining independent and dependent data
X = dataset.drop(columns=['Target','Target_name'])
Y = dataset['Target_name']

# Splitting data into test and train data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

# Standardize independent features
scaler = StandardScaler()
scaled = pd.DataFrame((scaler.fit_transform(X)), columns=X.columns)

# Implementing KNN algorithm
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, Y_train)
predicted = pd.DataFrame(model.predict(X_test))
category=['setosa', 'versicolor', 'virginica']
confusion_matrix = pd.DataFrame(confusion_matrix(Y_test, predicted), columns = category, index=category)
print(confusion_matrix)

# Saving the python code as Pickle File
pickle.dump(model, open('model.sav', 'wb'))