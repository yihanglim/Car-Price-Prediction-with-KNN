import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")

le = preprocessing.LabelEncoder()
price = le.fit_transform(list(data["price"]))  #Second hand price
age = le.fit_transform(list(data["age"]))      # car's age
rlb = le.fit_transform(list(data["reliable"]))  #reliability of the car, factors including maintenance cost, spare part availability
mile = le.fit_transform(list(data["mileage"])) #car's mileage
safety = le.fit_transform(list(data["safety"])) #safety feature of car
fc = le.fit_transform(list(data["fc"]))  #fuel consumption


predict = "price"

X = list(zip(years, age, rlb, mile, safety, fc))
y = list(price)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["very", "low", "high", "vhigh"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)