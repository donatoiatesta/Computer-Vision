import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

DATASET_PATH = "mnist/mnist.csv"

df = np.loadtxt(open(DATASET_PATH, "rb"), delimiter=",")

X = df[:, :-1]
y = df[:, -1:]

X, y = shuffle(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train /= 255
X_test /= 255

# trasformiamo i vettori con le classi in matrici binarie tramite il OneHotEncoder

y_train_dummy = to_categorical(y_train)
#y_test = to_categorical(y_test)

# creiamo la rete neurale

model = Sequential()

# aggiungiamo i nuovi strati
model.add(Dense(512, activation='relu', input_dim = X_train.shape[1]))
model.add(Dense(556, activation='relu'))
model.add(Dense(128, activation='relu'))
# aggiungiamo lo strato di output che deve avere il numero di neuroni pari al numero di classi all'interno del dataset
# come funzione di attivazione per lo stato di output utilizziamo la sigmoide
# Ma nel caso di classificazioni multiclasse la sigmoide considera come se ogni classe fosse
# indipendete l'una dall'altra
# La funzione da utilizzare in questi casi (Quando le classi sono dipendeti tra loro)
# è la funzione softamx
model.add(Dense(10, activation='softmax'))

# compiliamo il modello con l'algoritmo di ottimizzazione e la funzione di costo
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=["accuracy"])

model.fit(X_train, y_train_dummy, epochs=30, batch_size=512)

# ora eseguiamo le predizioni per poi calcolare le metriche
y_prob_train = model.predict(X_train)
y_pred_train = model.predict_classes(X_train)

y_prob_test = model.predict(X_test)
y_pred_test = model.predict_classes(X_test)

print("TRAIN_ACCURACY = %.4f"%accuracy_score(y_train, y_pred_train))
print("TRAIN_LOG_LOSS = %.4f"%log_loss(y_train, y_prob_train))

print("TEST_ACCURACY = %.4f"%accuracy_score(y_test, y_pred_test))
print("TEST_LOG_LOSS = %.4f"%log_loss(y_test, y_prob_test))

# salviamo il modello
model.save("model_mnist.h5")