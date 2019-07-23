import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

data = pd.read_csv("digit-recognizer/train.csv").as_matrix()

start = time.time()
#clf = DecisionTreeClassifier()
clf = KNeighborsClassifier()
# clf = RandomForestClassifier()
# clf = GaussianNB()
# clf = LogisticRegression()
# clf = LinearSVC()

# trainig dataset
xtrain = data[0:21000, 1:]
train_label = data[0:21000, 0]

clf.fit(xtrain, train_label)

print("a")
# testing data
xtest = data[21000:, 1:]
actual_label = data[21000:, 0]

d = xtest[101]
label = actual_label[101]
d.shape = (28, 28)
pt.imshow(255-d, cmap='gray')
print([label], clf.predict([xtest[101]]))
pt.show()

p = clf.predict(xtest)
print("b")
count = 0
for i in range(0, 21000):
    count += 1 if p[i] == actual_label[i] else 0
print("Uspjesnost:", (count/21000)*100)
end = time.time()
print("Vrijeme izvrsavanja:", end - start)