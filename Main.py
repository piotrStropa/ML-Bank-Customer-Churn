import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics
import seaborn as sns


dataset = pd.read_csv('Bank Customers Details.csv')
dataset.info()
X = dataset.drop('RowNumber', 1)
X = X.drop('CustomerId', 1)
X = X.drop('Surname', 1)
X = X.drop('Exited', 1)
y = dataset['Exited']

labelencoder_X_2 = LabelEncoder()
X['Gender'] = labelencoder_X_2.fit_transform(X['Gender'])
X = pd.get_dummies(X, columns=['Geography'],  drop_first=True)

# plt.figure(figsize=(20,20))
# churn_corr = dataset.corr()
# churn_corr_top = churn_corr.index
# sns.heatmap(dataset[churn_corr_top].corr(), annot=True)
# plt.show()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

models = [LogisticRegression(),GaussianNB(), KNeighborsClassifier(),
          SVC(probability=True),BaggingClassifier(),DecisionTreeClassifier(),
          RandomForestClassifier(), GradientBoostingClassifier()]
names = ["LogisticRegression","GaussianNB","KNN","SVC","Bagging",
             "DecisionTree","Random_Forest","GBM",]

print('Dokładność dla domyślnych modeli: ', end = "\n\n")
for name, model in zip(names, models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(name,':',"%.3f" % accuracy_score(y_pred, y_test))


classifier = keras.Sequential()

classifier.add(layers.Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(layers.Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(layers.Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 15, epochs = 250)

y_pred = classifier.predict(X_test)
y_pred = (y_pred >= 0.5)

cm = confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

