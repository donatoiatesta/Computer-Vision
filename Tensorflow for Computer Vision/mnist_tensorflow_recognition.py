# utilizziamo il modello creato per riconoscere nuove immagini
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# definiamo una costante per la dimensione dell'immagine essendo che il modello è stato 
# addestrato per immagini di dimensione 28x28
SCALE = (28, 28)

#importiamo il modello
model = load_model("model_mnist.h5")

img = cv2.imread("new_img_recognition/9.jpg", cv2.IMREAD_GRAYSCALE)

small_img = cv2.resize(img, SCALE)

# poniamo i pixel su un'unica riga quindi da 28x28 a 784

X = small_img.flatten().astype(float)
#normaiizziamo l'immagine
X /= 255.

# il modello non funziona bene perchè è stato addestrato su immagini con sfondo nero e numero bianco
# quindi convertiamo l'immagine invertendo i colori

X = 1.- X

# sklearn vuole in input per il metodo predict un array multidimensionale quindi bisogna
# aggiungere una dimensione al nostro array

X = X.reshape(1, X.shape[0])

# eseguiamo la predizione
pred = model.predict_classes(X)
proba = model.predict(X)
#pred = np.argmax(proba, axis = 1)
print("Model predict: %d con probabilità: %s" %(pred[0], np.round(proba[0], 3)))
cv2.imshow("number", X.reshape(28,28))
cv2.waitKey(0)
