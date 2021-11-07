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
nout = 4
activation = 'elu'

initializer = tf.keras.initializers.HeNormal()
x = Input(shape = (ndims,))
hidden = []
y = []
for i in range(nout):
    hidden.append(Dense(nodes, activation=activation, kernel_initializer = initializer)(x))
    for j in range(layers - 1):
        hidden[i] = Dense(nodes, activation=activation, kernel_initializer = initializer)(hidden[i])

for i in range(nout):
    y.append(Dense(1, activation = 'linear')(hidden[i]))
    
merge = tf.keras.layers.concatenate([y[0], y[1], y[2], y[3]])
merged = Model(inputs=x,outputs=merge)

merged.summary()
#plot_model(merged, show_shapes=True, show_layer_names=True)



x_test = np.random.uniform(0, 1, 100)
y_test = np.array([np.sin(x_test), np.cos(x_test), np.exp(x_test), np.sin(x_test)]).T
#print(np.shape(x))
#print(y.shape)
#print(y)



lr = 1e-3
loss = 'mae'
opt = Adam(lr)
merged.compile(optimizer = opt,
                  loss = loss)

print(merged.output_shape)
print(merged.input_shape)

merged.fit(x_test,y_test,  epochs = 10, batch_size = 10)












