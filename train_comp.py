import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import math
import pandas as pd
from datetime import datetime
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
flags.DEFINE_string('dir', 'ti4r4s', 'Directory',
                     short_name = 'dir')
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
flags.DEFINE_integer('n_train', 5*10**6 , '# of Training Pts',
                     short_name = 'n_train')
flags.DEFINE_integer('n_test', 5*10**6 , '# of Testing Pts',
                     short_name = 'n_test')
flags.DEFINE_integer('comp', 0 , 'Component',
                     short_name = 'comp')

#from kerastuner.tuners import RandomSearch
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
tf.keras.backend.set_floatx('float32')
print(tf.keras.backend.floatx())




##################### x_train #############################
#x_train = np.genfromtxt(x_train_path + "PVC_{}.csv".format(size_str), delimiter=',')
#y_train = np.genfromtxt(x_train_path + "PVC_label_{}.csv".format(size_str), delimiter = ',')


    #y_pred_less = y_pred_less.squeeze()


    #y_pred = forward_pass(model, x_test, scaler)
    #x_test_new = scaler_x_train.inverse_transform(x_test.reshape(-1,1))



################## Fit Model #################


def main(argv):
    del argv
######################################################
    nodes = FLAGS.nodes
    ndims = FLAGS.ndims
    nout = FLAGS.nout
    layers = FLAGS.layers
    epochs = FLAGS.epochs
    batch = FLAGS.batch
    loss = FLAGS.loss
    load = FLAGS.load
    real = FLAGS.real
    dir = FLAGS.dir
    n_train = FLAGS.n_train
    n_test = FLAGS.n_test
    comp = FLAGS.comp
######################################################
    norm = 'ss'
    str_ntrain = str(int(n_train/10**6))
    low = 1e-7
    high = 200
    parent = 'ti4r4s'
    dir_feat = 'ti4r4s'
    path = "./"+ dir +'/'
    size_str = '5M'
    size_str_test = '5M'
    ext_str = ''
    ext_str_test = ''
    extra = '_l{}h{}_{}M'.format(low, high, str_ntrain) ####### ADD UNDERSCORE!!
    N_func = 4
    lr = 1e-3
    activation = 'elu'
    cutoff = 1e-4
    N_func = 4
######################################################

    if load == 0:
        load = False
    elif load == 1:
        load = True

    path_features = path + 'data/'+ '{}_features_{}.txt'.format(dir, size_str)
    NAME = "{}_{}_nds_{}_lyr_{}_e_{}_b_{}_{}_{}_{}_{}{}".format(dir, comp, nodes, layers, epochs, batch, activation, loss, size_str, norm, extra)

    if real == 0:
        path_labels = path + '/data/'+ '{}_labels_{}_Re.txt'.format(dir, size_str)
        NAME = NAME + "_Re"
    elif real == 1:
        path_labels = path + '/data/'+ '{}_labels_{}_Im.txt'.format(dir, size_str)
        NAME = NAME + "_Im"




    x_test, y_test = cm.get_data_comp(path_features, path_labels, n_test, comp, low, high)
    if (not load):
        x_train, y_train = cm.get_data_comp(path_features, path_labels, n_train, comp, low, high)
        scaler_data, scaler = cm.scale(norm, path, NAME, low = -1000, high = 1000)
        x_train = scaler_data.fit_transform(x_train)
        print("This is y_train prefit = ", y_train)
        print("The max label = ", np.max(y_train))
        print("The min label = ", np.min(y_train))
        y_train = scaler.fit_transform(y_train.reshape(-1, 1)).squeeze()
        print("This is y_train = ", y_train)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.01, shuffle= True)
        print("The max label = ", np.max(y_train))
        print("The min label = ", np.min(y_train))
        pickle.dump(scaler, open(path + 'scaler/scaler_{}'.format(NAME), 'wb'))
        print("Dumped scaler")
    else:
        scaler = pickle.load(open(path + 'scaler/scaler_{}'.format(NAME), 'rb'))
        scaler_data =  FunctionTransformer()
        print("Scaler Loaded")
    print(NAME)



    count = 0
    if load == 0:
        while os.path.isfile(path + 'models/{}_{}.h5'.format(NAME, count)):
            count += 1
            if count > 100:
                break

    print("Count = ", count)
    print("This is y_test: ", y_test)
    print("The max is = ", np.max(y_test))
    print("The min is = ", np.min(y_test))





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

    #print(model(x_test))
    y_pred = cm.forward_pass(model, x_test, scaler)
    print(y_pred)
    abserr = np.abs(y_test - y_pred)
    errRe, args, idcs, means, mean_err, median_err,pts95_1, pts95_5, pts95_10, eta_1, eta_5, eta_10 = cm.get_measures(y_test, y_pred)
    ratio = np.divide(y_pred, y_test, out=np.ones_like(y_pred), where=np.logical_and(y_test!=0, y_pred!=0))
    #print(ratio[ratio<1e-7])
    logerr = np.log(np.abs(ratio))
    #plot.plot_2D(dir, NAME, abserr, x_test, 'Absolute Error', r'$\frac{t}{s}$', r'$\frac{m^2}{s}$', 'abserr', 'log')
    #plot.plot_2D(dir, NAME, y_pred, x_test, 'Neural Network', r'$\frac{t}{s}$', r'$\frac{m^2}{s}$', 'NN', 'symlog')
    # if (not load):
    #     df1 = pd.DataFrame({'Function': [dir],
    #                    'Layers': [layers],
    #                    'Nodes': [nodes],
    #                    'Activation': [activation],
    #                    'Epochs': [epochs],
    #                    'Batch': [batch],
    #                    'lr': [lr],
    #                    'real': [real],
    #                    'n_train': [n_train],
    #                    'n_test': [n_test],
    #                    'norm': [norm]
    #                    })
    #     for i in range(nout):
    #         df1["err{}".format(i)] = means[i]
    #     for i in range(nout, 22):
    #         df1["err{}".format(i)] = 'NaN'
    #     df1["Datetime"] = datetime.now()
    #     with open('run_data.csv', 'a') as f:
    #         f.write('\n')
    #     df1.to_csv('run_data.csv', mode='a', header = False, index = False)


    cm.plot_everything(path, dir, NAME, mean_err, median_err, eta_1, eta_5, eta_10, history, y_test, y_pred, idcs, errRe, x_test, logerr)

    #plot.plot_2D(dir, NAME, y_test[:,2], x_test, 'Actual', r'$\frac{t}{s}$', r'$\frac{m^2}{s}$', 'actual', 'symlog')
    #plot.plot_grid(path, dir, NAME, np.abs(errRe)*100, x_test, '% diff', r'$\frac{t}{s}$', r'$\frac{m^2}{s}$', 'relerrall', '', 'log')
    plot.plot_2D(path, dir, NAME, np.abs(errRe)*100, x_test, '% diff', r'$\frac{t}{s}$', r'$\frac{m^2}{s}$', 'relerr', str(comp), 'log')

    #for i in range(nout):
    #    plot.plot_2D(dir, NAME, np.abs(errRe[:,i])*100, x_test, '% diff', r'$\frac{t}{s}$', r'$\frac{m^2}{s}$', 'relerr', str(i), 'log')



if __name__ == '__main__':
    app.run(main)




    #execute(x_train_path, 'C010mimi0', size_str, size_str_test, ext_str, norm, loss, 2, 1)
    #execute(x_train_path, 'C0q0110', size_str, size_str_test, ext_str, norm, loss, 0, 1)
