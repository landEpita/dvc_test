from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten


import numpy as np
import sys
import mlflow.tensorflow
from atosflow.utils import *
mlflow.tensorflow.autolog()

from preprocessing import *
from sklearn.model_selection import train_test_split

def load_model():
    # load model without classifier layers
    model = VGG16(include_top=False, input_shape=(224, 224, 3))

    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(1024, activation='relu')(flat1)
    output = Dense(2, activation='softmax')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    return model

model = load_model()

labels = ["cats", "dogs"]
X,y = preprocessing(sys.argv[-1], labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

with mlflow.start_run() as run:
    run_uuid = run.info.run_uuid
    print("MLflow Run ID: %s" % run_uuid)
    model.fit(x_train,
             y_train,
             batch_size=64,
             epochs=1,
             validation_data=(X_test, y_test))

#compare(run_uuid,name='text')   
#print('fin')