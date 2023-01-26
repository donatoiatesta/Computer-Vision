# il modello creato nel vanilla neural network non era buono
# Vediamo come cambia con una rete convoluzionale

# Una matrice convoluzionale ha bisogno in input di una matrice di pixel e non di 
# un vettore come abbiamo fatto fin ora

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.layers import MaxPooling2D, Dropout

DATASET_FILENAME = "cat_dog.csv"
dataset = np.loadtxt(open(DATASET_FILENAME, "rb"), delimiter=",")

X = dataset[:, :-1]
y = dataset[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train /= 255.
X_test /= 255.

# ora dobbiamo riportare la dimensione dell'immagine a 64x64xnum canali dell'immagine
#utilizziamo un canale perchè stiamo lavorando in bainco e nero

X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
X_test = X_test.reshape(X_test.shape[0], 64, 64, 1)


model = Sequential()
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(64,64,1)))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Dropout(0.5))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Dropout(0.5))

# per far passare l'output di uno strato convoluzionale come input di uno strato denso 
# dobbiamo fare il fluttering un'altra volta, quindi bisogna passare da una rappresentazione
# a matrice ad una a vettori, quindi da 64x64x1 ad un vettore
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=50, batch_size=512)

# calcoliamo le metriche
metrics_train = model.evaluate(X_train, y_train, verbose=0)
metrics_test = model.evaluate(X_test, y_test, verbose=0)

print("Train Accuracy = %.4f - Train Loss = %.4f" % (metrics_train[1], metrics_train[0]))
print("Test Accuracy = %.4f - Test Loss = %.4f" % (metrics_test[1], metrics_test[0]))

model.save('conv_cat_dog.h5')