import numpy
from time import time
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

data = numpy.loadtxt(open("krkopt.data","rb"), delimiter=",", skiprows=0, dtype = int)

X = data[:,0:6]
Y = data[:,6]
seed = 7
max_features = 5
kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100

#Bagging
start = time()
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
end = time()
print(end - start)
print(results.mean())

#RandomForest
start = time()
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
end = time()
print(end - start)
print(results.mean())
