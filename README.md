# ML-Bank-Customer-Churn
Celem projektu jest zbudowanie i porównanie kilku modeli na podstawie danych zawierających klientów banku oraz informację o tym czy opuścili bank.
Dane pochodzą z https://www.kaggle.com/shrutimechlearn/churn-modelling
## Przetwarzanie danych
Opisy poszczególnych kolumn: 

```
RowNumber : Numer wiersza
CustomerId : Id klienta
Surname : Nazwisko klienta
CreditScore : The credit score of the customer
Geography : Kraj zamieszjania (Niemcy / Francja / Hiszpania)
Gender : Płeć
Age : Wiek
Tenure : Ile lat klient jest już w banku
Balance : Balans konta
NumOfProducts : Liczba produktów bankowych używanych przez klienta
HasCrCard : Czy klient posiada kartę
IsActiveMember : Czy klient jest aktywnym członkiem
EstimatedSalary : Pensja klienta
Exited : Czy klient opuścił bank
```




Zbiór danych zawiera następujące typy kolumn:
``` 

dataset.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 14 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   RowNumber        10000 non-null  int64  
 1   CustomerId       10000 non-null  int64  
 2   Surname          10000 non-null  object 
 3   CreditScore      10000 non-null  int64  
 4   Geography        10000 non-null  object 
 5   Gender           10000 non-null  object 
 6   Age              10000 non-null  int64  
 7   Tenure           10000 non-null  int64  
 8   Balance          10000 non-null  float64
 9   NumOfProducts    10000 non-null  int64  
 10  HasCrCard        10000 non-null  int64  
 11  IsActiveMember   10000 non-null  int64  
 12  EstimatedSalary  10000 non-null  float64
 13  Exited           10000 non-null  int64  
dtypes: float64(2), int64(9), object(3)
memory usage: 1.1+ MB
```
Jak widać większość kolumn zawiera typy numeryczne(int lub float).
Kolumny zawierające ciągi znaków to Surname, Geography oraz Gender.

Z góry możemy założyć, że trzy pierwsze kolumny (RowNumber, CustomerId oraz Surname) nie zawierają żadnych konkretnych danych mogących
wpłynąć na wynik, zatem zostaną one usunięte.

Kolumna Gender zawiera informacje o płci (2 wartości) więc do do zamiany na wartości liczbowe użyjemy LabelEncodera.

Kolumna Geography kategoryzuje klienta na podstawie jego kraju pochodzenia (3 wartości), więc aby nie zaburzyć żadnego modelu 
poprzez utworzenie sztucznej hierarchi/kolejności użyjemy OneHotEncodera (pd.getDummies)

Z funkcji info możemy się dowiedzieć iż żadna kolumna nie zawiera wartości nullowych (10k wierszy i na każdej kolumnie 10k non-null)

```python
X = dataset.drop('RowNumber', 1)
X = X.drop('CustomerId', 1)
X = X.drop('Surname', 1)
X = X.drop('Exited', 1)
y = dataset['Exited']

labelencoder_X_2 = LabelEncoder()
X['Gender'] = labelencoder_X_2.fit_transform(X['Gender'])
X = pd.get_dummies(X, columns=['Geography'],  drop_first=True)

```

Następnie po przetworzeniu wszystkich danych na numeryczne możemy przejść do podzielenia 
danych na train i test.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

Następnie dane zostały znormalizowane przy użyciu StandardScalera.
```python
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
Macierz korelacji wygląda następująco:
```python
plt.figure(figsize=(20,20))
churn_corr = dataset.corr()
churn_corr_top = churn_corr.index
sns.heatmap(dataset[churn_corr_top].corr(), annot=True)
plt.show()
```
![Macierz korelacji](https://github.com/piotrStropa/ML-Bank-Customer-Churn/blob/main/corr.png?raw=true)


## Modele
Zacznijmy od sprawdzenia domyślnych modeli:
- LogisticRegression
- GaussianNB
- KNeighborClassifier
- SVC
- BaggingClassifier
- RandomForestClassifier
- GradientBoostingClassifier

```python
models = [LogisticRegression(),GaussianNB(), KNeighborsClassifier(),
          SVC(probability=True),BaggingClassifier(),DecisionTreeClassifier(),
          RandomForestClassifier(), GradientBoostingClassifier()]
names = ["LogisticRegression","GaussianNB","KNN","SVC","Bagging",
             "DecisionTree","Random_Forest","GBM",]

print('Dokładność dla domyślnych modeli: ', end = "\n\n")
for name, model in zip(names, models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(name,':',"%.3f" % accuracy_score(y_pred, y_test))```
```
Wyniki dla poszczególnych modeli: 

```python
LogisticRegression : 0.811
GaussianNB : 0.830
KNC : 0.827
SVC : 0.864
Bagging : 0.859
DecisionTree : 0.803
Random_Forest : 0.869
GBC : 0.870
```
Jak widać najlepiej wypadł RandomForestClassifier oraz GradientBoostingClassifier

# Sieć neuronowa

Do zbudowania sieci neuronowej został użyty model sekwencyjny który jest zbudowany przy założeniu, 
że dane wejściowe następnej warstwy są wyjściami poprzedniej warstwy. 

Każda warstwa jest warstwą Dense (gęstą) tzn.  wszystkie jednostki poprzedniej warstwy są połączone ze wszystkimi w następnej,
dwie pierwsze warstwy mają funkcję aktywacji 'relu'.

Rozważamy problem klasyfikacji binarnej więc końcowo najlepiej jest przyjąć funkcję aktywacji w kształcie 'S' (sigmoid).

Następnie wykonujemy 250 epok trenowania modelu. 

```python
classifier = keras.Sequential()

classifier.add(layers.Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(layers.Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(layers.Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 15, epochs = 100)

```

```python
Epoch 250/250
8000/8000 [==============================] - 0s 53us/sample - loss: 0.3208 - acc: 0.8708
```
Wynik accuracy wskauje, że zamodelowana w ten sposób sieć neuronowa, daje lepsze wyniki niż domyślne modele.

Macierz błędu wygląda następująco:
```python
y_pred = classifier.predict(X_test)
y_pred = (y_pred >= 0.5)

cm = confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
```

![Macierz błędu](https://github.com/piotrStropa/ML-Bank-Customer-Churn/blob/main/confusion.png?raw=true)



