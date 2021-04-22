import numpy as np
from os import listdir
from os.path import isfile, join
import sys
import cv2

from keras.applications.vgg16 import preprocess_input
#from keras.applications.vgg16 import decode_predictions

def load_data(path,label):
    X = []
    Y = []
    i = 0
    for label in labels:
        current_path = join(path,label)
        onlyfiles = [join(current_path, f) for f in listdir(current_path) if isfile(join(current_path, f))]
        for f in onlyfiles:
            im = cv2.imread(f)
            im = cv2.resize(im, (224,224))
            X.append(im)
            Y.append(i)
        i+=1
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def preprocessing(path,labels):
    x,y = load_data(path,labels)
    X = preprocess_input(x)
    return X,y
