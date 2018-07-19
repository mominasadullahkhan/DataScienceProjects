# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential # Neural Network
from keras import optimizers # Neural Network optimizer
from keras.optimizers import RMSprop, SGD, Adam # Optimizers
from keras.utils import np_utils # Neural Network
from keras.layers.core import Dense, Activation, Dropout # Neural Network
import matplotlib.pyplot as plt # Visualization


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Load the data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.head()
x = df_train.drop(['label'],axis=1)
y = df_train['label']
y = pd.get_dummies(y)

# Preprocess the digit data : divide by max and subtract by mean

# Get input dimensions and classes
x.shape
y.shape
i_dimensions = x.shape[1]
n_c = y.shape[1]
i_dimensions
X_train = x.astype('float32')
X_test = df_test.astype('float32')
xtt = df_test.astype('float32')

X_train = X_train.values.reshape(X_train.shape[0], 784).astype('float32')
X_test = X_test.values.reshape(X_test.shape[0], 784).astype('float32')

# Normalize Inputs
X_train = X_train / 255
X_test = X_test / 255

# Build the model
model = Sequential()
model.add(Dense(i_dimensions, input_dim=i_dimensions, kernel_initializer='normal', activation='relu'))
model.add(Dense(n_c, kernel_initializer='normal', activation='softmax'))

# Compile model
sgd = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
print("Training...")
model.fit(X_train, y, nb_epoch=20, batch_size=128, validation_split=0.1, verbose=2)

# Make predictions on test set
print("Generating test predictions...")
pred = model.predict(X_test)
# Evaluate model on traning set
eva = model.evaluate(X_train,y)
# Returns the digit predicted
ee = np.argmax(pred, axis=1)

#Prediction using model.predict_classes
m_pred = model.predict_classes(X_test, batch_size=32, verbose=1)


def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(ee, "DigitRecognizer.csv")
