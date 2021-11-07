import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import math
import pandas as pd
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
flags.DEFINE_integer('real', 0 , 'Real or Imag',
                     short_name = 'real')

#from kerastuner.tuners import RandomSearch
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
tf.keras.backend.set_floatx('float32')
print(tf.keras.backend.floatx())




norm = 'ss'
#scaler_data = MinMaxScaler((-1e3,1e3))
#scaler_data = FunctionTransformer(expp5, inverse_func = logm5)
#scaler_data = StandardScaler()
#scaler_data = FunctionTransformer()

str_ntrain = '5'
low = 1e-3
high = 200
parent = 'ti4r1t'
C_str_feat = 'ti4r1t'
C_str = 'ti4r1t'
path = "./"
data_path = path  + 'data/'
size_str = '5M'
n_train = int(str_ntrain)*10**6
n_test = int(str_ntrain)*10**6
size_str_test = '5M'
ext_str = ''
ext_str_test = ''
extra = '_l{}h{}_{}M'.format(low, high, str_ntrain) ####### ADD UNDERSCORE!!


N_func = 4
lr = 1e-3
activation = 'elu'


cutoff = 1e-4

##################### Data #############################
#x_train = np.genfromtxt(data_path + "PVC_{}.csv".format(size_str), delimiter=',')
#y_train = np.genfromtxt(data_path + "PVC_label_{}.csv".format(size_str), delimiter = ',')


def execute(data_path, C_str_feat, C_str, size_str, size_str_test, ext_str ,norm, loss, idx, idx_y,
            ndims, nout, nodes, layers, activation, epochs, batch, load, n_train, n_test):

    data, labels = cm.get_data(data_path, C_str_feat, C_str, size_str, ext_str, idx, idx_y, n_train, low, high)
    x_test, y_test = cm.get_test_data(data_path, C_str_feat, C_str, size_str_test,ext_str_test, idx, idx_y, n_test, low, high)

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
    y_train = labels
    count = 0
    if load == 0:
        while os.path.isfile(path + 'models/{}_{}.h5'.format(NAME, count)):
            count += 1
            if count > 100:
                break

    #x_test, y_test = get_test_data(path + 'ti4r1t' +'_data/', 'ti4r1t','C010mimi0', size_str_test,ext_str_test, idx)
    print("Count = ", count)

    data = scaler_data.fit_transform(data)
    #x_test_new, y_pred = sort(x_test_new, y_pred)
    #x_test, y_test = sort(x_test, y_test)
    print("This is y_test: ", y_test)
    print("The max is = ", np.max(y_test))
    print("The min is = ", np.min(y_test))
    print("The max label = ", np.max(y_train))
    print("The min label = ", np.min(y_train))




    if norm == 'mm':
    	scaler = MinMaxScaler((-1, 1))
    	y_train = scaler.fit_transform(y_train)
    elif norm == 'ss':
    	scaler = StandardScaler()
    	y_train = scaler.fit_transform(y_train)
    elif norm == 'log':
    	scaler = FunctionTransformer(np.log1p, inverse_func = np.expm1)
    	y_train = scaler.transform(y_train + np.abs(ymin))
    else:
        scaler = FunctionTransformer()


    pickle.dump(scaler, open(path + 'scaler/scaler_{}'.format(NAME), 'wb'))
    print("Dumped MinMax scaler")



    ###################### Data split ###########################
    x_train, x_valid, y_train, y_valid = train_test_split(data, y_train, test_size=0.01, shuffle= True)


    #layer = Dense(nodes, activation=activation, kernel_initializer = initializer)

    ##########################################



    if load:
        model = tf.keras.models.load_model(path + 'models/{}_'.format(NAME) + str(count) + '.h5', compile = False)
        history = pickle.load(open(path + 'history/history_{}'.format(NAME), "rb"))
        print("Model has loaded")
    else:

        checkpoint_filepath = path + 'tmp/checkpoint_{}'.format(NAME)
        ckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='loss',
            mode='auto',
            save_best_only=True)

        csv_logger = CSVLogger(path + 'logs/training_{}.log'.format(NAME))
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                  patience=epochs//25, min_lr=1e-8, verbose = 1)


        stop = EarlyStopping(
            monitor='val_loss', patience=500, verbose=0,
            mode='auto', baseline=None, restore_best_weights=True
            )

        model = cm.init_model(ndims, nout, nodes, layers, activation, loss)
        model.save(path + 'models/{}_'.format(NAME) + str(count) + '.h5')
        history = model.fit(x_train, y_train, validation_data = (x_valid, y_valid),
    		        epochs = epochs, batch_size = batch, shuffle = True,
    		        callbacks = [reduce_lr, ckpt, csv_logger], verbose = 1)
        model.load_weights(checkpoint_filepath)
        print("Model has been fit")
        model.save(path + 'models/{}_'.format(NAME) + str(count) + '.h5')
        print("Model Saved")
        with open(path + 'history/history_{}'.format(NAME), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        print("History Saved")


    y_pred = []
    """
    x_split, y_split = split(x_test, y_test, cutoff, idx)

    #x_test = scaler_data.transform(x_test.reshape(-1,1))

    for i in range(N_func):
        print("x_{} = ".format(i+1), x_split[i])
        print("y_{} = ".format(i+1), y_split[i])


    x_split[0]= forward_transform(x_split[0], scaler_data)
    y_pred.append(forward_pass(model, x_split[0], scaler))
    x_split[0]= inverse_transform(x_split[0], scaler_data)

    start = time.time()
    if idx==0:
        for i in range(N_func - 1):
            y_pred.append(C0.C010rs0(x_split[i+1], idx_y, i+1))
    elif idx == 1:
        #if idx_y == 0:
        #    y_pred_great = remove_zero(x_great, y_pred_great, 1e-3)
        for i in range(N_func - 1):
            y_pred.append(C0.C0n10rs0(x_split[i+1], idx_y, i+1))
    elif idx == 2:
        for i in range(N_func - 1):
            y_pred.append(C0.C0r01s0(x_split[i+1], idx_y, i+1))
    elif idx == 3:
        for i in range(N_func - 1):
            y_pred.append(C0.C0r0s10(x_split[i+1], idx_y, i+1))
    end = time.time() - start
    print("Time to execute analytic pass = ", end, " seconds")

    y_pred_all = np.concatenate((y_pred[0], y_pred[1], y_pred[2], y_pred[3]))
    y_test_all = np.concatenate((y_split[0], y_split[1], y_split[2], y_split[3]))
    #y_test = np.concatenate((y_less, y_great))
    x_test_all = np.concatenate((x_split[0], x_split[1], x_split[2], x_split[3]))
    print("This is ypred = ", y_pred_all[:100])
    print("This is ypred = ", y_test_all[:100])
    errRe, args, idcs, mean_err, median_err,pts95_1, pts95_5, pts95_10, eta_1, eta_5, eta_10 = get_measures(y_test_all, y_pred_all)
    err_1, _, _, _, _, _, _, _, _, _, _ = get_measures(y_split[1], y_pred[1])
    ratio = np.divide(y_pred_all, y_test_all, out=np.ones_like(y_pred_all), where=np.logical_and(y_test_all!=0, y_pred_all!=0))
    #print(ratio[ratio<1e-7])
    logerr = np.log(np.abs(ratio))

    plot_everything(NAME, mean_err, median_err, eta_1, eta_5, eta_10, history, y_test_all, y_pred_all, idcs, errRe, x_test_all, logerr)
    """



    y_pred = cm.forward_pass(model, x_test, scaler)
    print(y_pred)
    abserr = np.abs(y_test - y_pred)
    errRe, args, idcs, means, mean_err, median_err,pts95_1, pts95_5, pts95_10, eta_1, eta_5, eta_10 = cm.get_measures(y_test, y_pred)
    ratio = np.divide(y_pred, y_test, out=np.ones_like(y_pred), where=np.logical_and(y_test!=0, y_pred!=0))
    #print(ratio[ratio<1e-7])
    logerr = np.log(np.abs(ratio))
    #plot.plot_2D(C_str, NAME, abserr, x_test, 'Absolute Error', r'$\frac{t}{s}$', r'$\frac{m^2}{s}$', 'abserr', 'log')
    #plot.plot_2D(C_str, NAME, y_pred, x_test, 'Neural Network', r'$\frac{t}{s}$', r'$\frac{m^2}{s}$', 'NN', 'symlog')

    df1 = pd.DataFrame({'Function': [C_str],
                   'Layers': [layers],
                   'Nodes': [nodes],
                   'Activation': [activation],
                   'Epochs': [epochs],
                   'lr': [lr],
                   })
    for i in range(nout):
        df1["err{}".format(i)] = means[i]
    with open('run_data.csv', 'a') as f:
        f.write('\n')
    df1.to_csv('run_data.csv', mode='a', header=False, index = False)
    #plot.plot_2D(C_str, NAME, y_test[:,2], x_test, 'Actual', r'$\frac{t}{s}$', r'$\frac{m^2}{s}$', 'actual', 'symlog')
    for i in range(nout):
        plot.plot_2D(C_str, NAME, np.abs(errRe[:,i])*100, x_test, '% diff', r'$\frac{t}{s}$', r'$\frac{m^2}{s}$', 'relerr', str(i), 'log')

    cm.plot_everything(C_str, NAME, mean_err, median_err, eta_1, eta_5, eta_10, history, y_test, y_pred, idcs, errRe, x_test, logerr)

    #y_pred_less = y_pred_less.squeeze()


    #y_pred = forward_pass(model, x_test, scaler)
    #x_test_new = scaler_data.inverse_transform(x_test.reshape(-1,1))



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
    real = FLAGS.real



    if load == 0:
        load = False
    elif load == 1:
        load = True

    N_func = 4
    func_string=['ti4r1t', 'ti4r1t', 'ti4r1t']

    """
    for i in range(N_func):
        for j in range(2):
            execute(data_path, C_str_feat, func_string[i], size_str, size_str_test, ext_str, norm, loss, i, j,
                    ndims, nodes, layers, activation, epochs, batch, load))
    """

    execute(data_path, func_string[0], func_string[0], size_str, size_str_test, ext_str, norm, loss, 0, real,
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
