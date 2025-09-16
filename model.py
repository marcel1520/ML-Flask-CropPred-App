import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv("Crop_recommendation.csv")

x = data.iloc[:, : -1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

model = RandomForestClassifier()

model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))

