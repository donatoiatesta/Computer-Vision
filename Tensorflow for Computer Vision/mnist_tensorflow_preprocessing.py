import cv2
import os

DATASET_PATH = "mnist/"

DATASET_FILENAME = "mnist.csv"

data_file = open(DATASET_PATH+DATASET_FILENAME, "w")

counter = [0]*10

for i in range(10):
    current_dir = DATASET_PATH+str(i)

    for f in os.listdir(current_dir):

        #assicuriamoci che sia un'immagini
        #controlliamo che il file abbia estensione jpg
        if(".jpg" not in f):
            continue
        
        img = cv2.imread(current_dir+"/"+f, cv2.IMREAD_GRAYSCALE)

        # ora ridimensioniamo l'immagine
        arr = img.flatten()

        # scriviamo l'array nel file csv
        arr_str = ','.join(arr.astype(str))
        data_file.write(arr_str)

        data_file.write(","+str(i))
        data_file.write("\n")

        counter[i] += 1

data_file.close()
print("Immagini scritte per classe: %s" %counter) 