import os
import cv2

DATASET_PATH = "cat_dog_small/"
OUT_FILE = "cat_dog.csv"
SCALE = (64, 64)

out = open(OUT_FILE, "w")

# creiamo un dizionario per mappare le classi ai label numerici

classes = {'cat': "1", "dog": "0"}
counter = {'cat': 0, 'dog': 1}

print("Lettura di tutte le immagini da %s" %DATASET_PATH)
print("Scrittura in %s" % OUT_FILE)

for c in classes:
    current_dir = DATASET_PATH+c
    for f in os.listdir(current_dir):
        if(not ".jpg" in f):
            continue

        img = cv2.imread(current_dir+'/'+f, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, SCALE)
        img = img.flatten().astype(str)
        data = ",".join(img)+","+classes[c]
        out.write(data+'\n')
        
        counter[c] += 1

    
out.close()
print("Immagini scritte: %s"%counter)