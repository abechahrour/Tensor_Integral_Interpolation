import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import math
from matplotlib import gridspec
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback, CSVLogger
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K
import linecache
import tracemalloc
from absl import app, flags
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import common as cm
import plot


FLAGS = flags.FLAGS

flags.DEFINE_string('loss', 'mse', 'The loss function',
                     short_name = 'l')
flags.DEFINE_integer('ndims', 2, 'The number of dimensions',
                     short_name='d')
flags.DEFINE_integer('nout', 1, 'The number of outputs',
                     short_name='nout')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train',
                     short_name='e')
flags.DEFINE_integer('batch', 100000, 'Number of points to sample per epoch',
                     short_name='b')
flags.DEFINE_integer('nodes', 100, 'Num of nodes',
                     short_name = 'nd')
flags.DEFINE_integer('layers', 8 , 'Exponent of cut function',
                     short_name = 'lyr')
flags.DEFINE_integer('load', 0 , 'Load Model',
                     short_name = 'load')

#from kerastuner.tuners import RandomSearch
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
tf.keras.backend.set_floatx('float32')
print(tf.keras.backend.floatx())




norm = 'mm'

str_ntrain = '5'
low = 1e-3
high = 200
parent = 'ti4r1s'
C_str_feat = 'ti4r1s'
C_str = 'ti4r1s'
path = "./"
data_path = path  + 'data/'
size_str = '5M'
n_train = int(str_ntrain)*10**6
n_test = int(str_ntrain)*10**6
size_str_test = '5M'
ext_str = ''
ext_str_test = ''
#extra = '_n1to1_l{:.0e}h{}_{}M'.format(low, high, str_ntrain) ####### ADD UNDERSCORE!!
extra = '_n1to1_l{}h{}_{}M'.format(low, high, str_ntrain) ####### ADD UNDERSCORE!!


N_func = 4
lr = 1e-3
activation = 'elu'


cutoff = 1e-4

##################### Data #############################
#x_train = np.genfromtxt(data_path + "PVC_{}.csv".format(size_str), delimiter=',')
#y_train = np.genfromtxt(data_path + "PVC_label_{}.csv".format(size_str), delimiter = ',')


def execute(data_path, C_str_feat, C_str, size_str, size_str_test, ext_str ,norm, loss, idx, idx_y,
            ndims, nout, nodes, layers, activation, epochs, batch, load, n_train, n_test):

    x = np.random.uniform([-1, 0], [0, 0.25], (10**6, 2))

    if idx == 0 or idx == 1:
        #scaler_data_1 = MinMaxScaler((1e-3,1e2))
        #scaler_data_2 = MinMaxScaler((-1e2, 1e2))
        #data1 = scaler_data_1.fit_transform(data[1])
        #data2 = scaler_data_2.fit_transform(data[0])
        #data = np.column_stack((data1,data2))
        scaler_data = FunctionTransformer()
    elif idx == 2 or idx == 3:
        #scaler_data = MinMaxScaler((-1e2,1e2))
        scaler_data = FunctionTransformer()


    #ndims = int(np.shape(data)[1]/)
    if idx_y == 0:
        NAME = "{}_nds_{}_lyr_{}_e_{}_b_{}_{}_{}_{}_{}{}_Re".format(C_str,nodes, layers, epochs, batch, activation, loss, size_str, norm, extra)
    elif idx_y == 1:
        NAME = "{}_nds_{}_lyr_{}_e_{}_b_{}_{}_{}_{}_{}{}_Im".format(C_str,nodes, layers, epochs, batch, activation, loss, size_str, norm, extra)
    print(NAME)
    #y_train = labels
    count = 0


    #x, y_test = get_test_data(path + 'ti4r1s' +'_data/', 'ti4r1s','C010mimi0', size_str_test,ext_str_test, idx)
    print("Count = ", count)

    #data = scaler_data.fit_transform(data)



    scaler = pickle.load(open(path + 'scaler/scaler_{}'.format(NAME), 'rb'))
    print("Loaded MinMax scaler")


    model = tf.keras.models.load_model(path + 'models/{}_'.format(NAME) + str(count) + '.h5', compile = False)
    history = pickle.load(open(path + 'history/history_{}'.format(NAME), "rb"))
    print("Model has loaded")


    metric = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
    p = np.array([0.5, 0, 0., 0.5])
    p = np.einsum('i, ij', p, metric)
    y_pred = []
    y_pred = cm.forward_pass(model, x, scaler)
    y_pred = np.insert(y_pred, 1, 0, axis=1)
    res = np.tensordot(y_pred, p, 1)
    print("Input = ", x)
    print("Output = ", res)

    # print(y_pred)
    # abserr = np.abs(y_test - y_pred)
    # errRe, args, idcs, mean_err, median_err,pts95_1, pts95_5, pts95_10, eta_1, eta_5, eta_10 = cm.get_measures(y_test, y_pred)
    # ratio = np.divide(y_pred, y_test, out=np.ones_like(y_pred), where=np.logical_and(y_test!=0, y_pred!=0))
    # #print(ratio[ratio<1e-7])
    # logerr = np.log(np.abs(ratio))
    # #plot.plot_2D(C_str, NAME, abserr, x, 'Absolute Error', r'$\frac{t}{s}$', r'$\frac{m^2}{s}$', 'abserr', 'log')
    # #plot.plot_2D(C_str, NAME, y_pred, x, 'Neural Network', r'$\frac{t}{s}$', r'$\frac{m^2}{s}$', 'NN', 'symlog')
    # for i in range(nout):
    #     plot.plot_2D(C_str, NAME+'_{}'.format(i), y_test[:,i], x, 'Actual', r'$\frac{t}{s}$', r'$\frac{m^2}{s}$', 'actual', 'symlog')
    #     plot.plot_2D(C_str, NAME, np.abs(errRe[:,i])*100, x, '% diff', r'$\frac{t}{s}$', r'$\frac{m^2}{s}$', 'relerr', str(i), 'log')
    #
    # cm.plot_everything(C_str, NAME, mean_err, median_err, eta_1, eta_5, eta_10, history, y_test, y_pred, idcs, errRe, x, logerr)

    #y_pred_less = y_pred_less.squeeze()


    #y_pred = forward_pass(model, x, scaler)
    #x_new = scaler_data.inverse_transform(x.reshape(-1,1))



################## Fit Model #################


def main(argv):
    del argv

    nodes = FLAGS.nodes
    ndims = FLAGS.ndims
    nout = FLAGS.nout
    layers = FLAGS.layers
    epochs = FLAGS.epochs
    batch = FLAGS.batch
    loss = FLAGS.loss
    load = FLAGS.load

    if load == 0:
        load = False
    elif load == 1:
        load = True

    N_func = 4
    func_string=['ti4r1s', 'ti4r1s', 'ti4r1s']

    """
    for i in range(N_func):
        for j in range(2):
            execute(data_path, C_str_feat, func_string[i], size_str, size_str_test, ext_str, norm, loss, i, j,
                    ndims, nodes, layers, activation, epochs, batch, load))
    """

    execute(data_path, func_string[0], func_string[0], size_str, size_str_test, ext_str, norm, loss, 0, 0,
            ndims,nout, nodes, layers, activation, epochs, batch, load, n_train, n_test)

    #execute(data_path, func_string[0], func_string[0], size_str, size_str_test, ext_str, norm, loss, 0, 1,
    #        ndims, nodes, layers, activation, epochs, batch, load)

    #execute(data_path, 'rs'+ func_string[0], 'rs'+func_string[0], size_str, size_str_test, ext_str, norm, loss, 0, 0,
    #        ndims, nodes, layers, activation, epochs, batch, load)

    #execute(data_path, func_string[1], func_string[1], size_str, size_str_test, ext_str, norm, loss, 0, 0,
    #        ndims, nodes, layers, activation, epochs, batch, load)

    #execute(data_path, 'rs'+ func_string[1], 'rs'+func_string[1], size_str, size_str_test, ext_str, norm, loss, 0, 0,
    #        ndims, nodes, layers, activation, epochs, batch, load)

    #execute(data_path, C_str_feat, func_string[2], size_str, size_str_test, ext_str, norm, loss, 2, 0,
    #        ndims, nodes, layers, activation, epochs, batch, load)

    #execute(data_path, C_str_feat, func_string[3], size_str, size_str_test, ext_str, norm, loss, 0, 0,
    #        ndims, nodes, layers, activation, epochs, batch, load))


if __name__ == '__main__':
    app.run(main)




    #execute(data_path, 'C010mimi0', size_str, size_str_test, ext_str, norm, loss, 2, 1)
    #execute(data_path, 'C0q0110', size_str, size_str_test, ext_str, norm, loss, 0, 1)
