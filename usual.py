import pickle
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import itertools
from scipy.spatial import distance

from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback, CSVLogger
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model




ndims = 1
nodes = 32
layers = 2
nout = 1
activation = 'elu'
lr = 1e-3
loss = 'mse'

initializer = tf.keras.initializers.HeNormal()
x = Input(shape = (ndims,))
h = Dense(nodes, activation=activation, kernel_initializer = initializer)(x)
#h = BatchNormalization()(h)
for i in range(layers - 1):
    h = Dense(nodes, activation=activation, kernel_initializer = initializer)(h)
    #h = BatchNormalization()(h)

y = Dense(nout, activation = 'linear')(h)
model = Model(x, y)
model.summary()
print("The Learning Rate is  ", lr)
opt = Adam(lr)
model.compile(optimizer = opt,
              loss = loss)

x_train = [1, 2, 3, 4, 5, 6]
x_train = np.random.uniform(0, 1, 100)
y_train = x_train
#print(x_train.shape)
#print(y_train.shape)
model.fit(x_train, y_train, epochs = 10, batch_size = 10)


