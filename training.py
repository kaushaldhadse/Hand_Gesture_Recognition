import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

datad = pickle.load(open('C:\Python programming\HandGestureDetection\pikolo.pickle', 'rb'))

data = np.array(datad['data'])
labels = np.array(datad['labels'])

X = data
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

acs = accuracy_score(y_predict, y_test)

print("Accuracy = " + str(acs*100) + " %")


ff = open('C:\Python programming\HandGestureDetection\model.p', 'wb')
pickle.dump({'model' : model, 'labels': labels}, ff)
ff.close()