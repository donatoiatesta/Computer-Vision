import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

DATASET_FILENAME = "cat_dog.csv"
dataset = np.loadtxt(open(DATASET_FILENAME, "rb"), delimiter=",")

X = dataset[:, :-1]
y = dataset[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train /= 255.
X_test /= 255.

# essendo una classificazione binaria non dobbiamo eseguire il OneHotEncoder

model = Sequential()
model.add(Dense(512, activation='relu', input_dim = X_train.shape[1]))
model.add(Dense(556, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=30, batch_size=512)

# calcoliamo le metriche
metrics_train = model.evaluate(X_train, y_train, verbose=0)
metrics_test = model.evaluate(X_test, y_test, verbose=0)

print("Train Accuracy = %.4f - Train Loss = %.4f" % (metrics_train[1], metrics_train[0]))
print("Test Accuracy = %.4f - Test Loss = %.4f" % (metrics_test[1], metrics_test[0]))
