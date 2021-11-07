import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
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
import plot
import linecache
import tracemalloc
from absl import app, flags
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


def split(x, y, eps,idx, train = False):
    x = x.squeeze()
    print("Shape of x in split ", x.shape)
    ratio = 0.5
    eps_tr = eps*0.1
    if idx ==0 or idx == 1:
        s = x[:,1]
    else:
        s = x[:,1]**2
        eps = eps**2
        eps_tr = eps_tr**2

    r = x[:,0]

    if train:
        print("Test is False")
        cond = np.where(np.abs(s) < eps_tr, True, False)
        print("Shape of cond: ", cond.shape)
        loc = np.where(np.abs(r) < eps_tr,
                    np.where(cond,
                            np.where(np.abs(r/s) < ratio, '1',
                                    np.where(np.abs(s/r) < ratio,'2', '3')),
                            '1'),
                    np.where(cond, '2', 'NN'))
    else:
        print("Test is true")
        cond = np.where(np.abs(s) < eps, True, False)
        loc = np.where(np.abs(r) < eps,
                    np.where(cond,
                            np.where(np.abs(r/s) < ratio, '1',
                                    np.where(np.abs(s/r) < ratio,'2', '3')),
                            '1'),
                    np.where(cond, '2', 'NN'))
    idc_NN = np.where(loc=='NN')
    idc_r  = np.where(loc=='1')
    idc_s  = np.where(loc=='2')
    idc_rs = np.where(loc=='3')
    x_NN = x[idc_NN]
    y_NN = y[idc_NN]
    x_r = x[idc_r]
    y_r = y[idc_r]
    x_s = x[idc_s]
    y_s = y[idc_s]
    x_rs = x[idc_rs]
    y_rs = y[idc_rs]
    x = [x_NN, x_r, x_s, x_rs]
    y = [y_NN, y_r, y_s, y_rs]

    return x, y

def remove_zero(x, y, eps):
    idcs = np.where(np.abs(x - 0.281772) < eps)
    y[idcs] = 0.
    return y

def sort(x, y):
    print("Shape of x in sort: ", x.shape)
    idcs = np.argsort(x)
    print(x[idcs[-3]])
    print(y[idcs[-3]])
    print(x[idcs[2]])
    print(y[idcs[2]])
    print("The idcs in sort are: ", idcs)
    x = np.sort(x)

    y = y[idcs]
    print("The y in sort is: ", y)
    return x, y
def forward_transform(x, scaler):
    return scaler.transform(x).squeeze()

def inverse_transform(x, scaler):
    return scaler.inverse_transform(x).squeeze()



norm = 'mm'
#scaler_data = MinMaxScaler((-1e3,1e3))
#scaler_data = FunctionTransformer(expp5, inverse_func = logm5)
#scaler_data = StandardScaler()
#scaler_data = FunctionTransformer()

parent = 'ti3r1'
C_str_feat = 'ti3r1'
C_str = 'ti3r1'
path = '/home/chahrour/golem_NN/' + parent + '/'
data_path = path + C_str_feat +'_data/'
size_str = '1M'
n_train = 5*10**6
n_test = 5*10**6
size_str_test = '1M'
ext_str = ''
ext_str_test = ''
extra = '_n1to1' ####### ADD UNDERSCORE!!


N_func = 4
lr = 1e-3
activation = 'elu'


cutoff = 1e-4

##################### Data #############################
#x_train = np.genfromtxt(data_path + "PVC_{}.csv".format(size_str), delimiter=',')
#y_train = np.genfromtxt(data_path + "PVC_label_{}.csv".format(size_str), delimiter = ',')


def get_data(data_path , C_str_feat, C_str, size_str, ext_str, idx, idx_y):

    print("Loading Data")
    x = np.genfromtxt(data_path + "{}_features_{}{}.txt".format(C_str_feat, size_str, ext_str), delimiter=',', max_rows = n_train)
    if idx_y==0:
        y = np.genfromtxt(data_path + "{}_labels_{}{}_Re.txt".format(C_str, size_str, ext_str), delimiter = ',', max_rows = n_train)
    elif idx_y == 1:
        y = np.genfromtxt(data_path + "{}_labels_{}{}_Im.txt".format(C_str, size_str, ext_str), delimiter = ',', max_rows = n_train)
    print("Data Loaded")
    print("Shape of x_train: ", x.shape)
    x = x[:,:]
    y = y[:,[0, 3]]
    #idcs = np.where(np.logical_and(np.abs(y[:,idx_y]) < 200,np.abs(y[:,idx_y]) > 1e-8))
    #idcs = np.where(np.logical_and(x[:,0] < 0, x[:,1] < 0.5))
    #x = x[idcs]
    #y = y[idcs]
    #print("Max of y_train is ", np.max(y))
    #print("Min of y_train is ", np.min(y))
    #x, y = split(x, y, cutoff, idx, True)
    #del x_train, y_train
    #print(x_train[np.where(x_train < 1e-7)].shape)

    return x, y

def get_test_data(data_path , C_str_feat, C_str, size_str, ext_str, idx, idx_y):
    x = np.genfromtxt(data_path + "{}_features_{}{}.txt".format(C_str_feat, size_str, ext_str), delimiter=',', max_rows = n_test)
    if idx_y==0:
        y = np.genfromtxt(data_path + "{}_labels_{}{}_Re.txt".format(C_str, size_str, ext_str), delimiter = ',', max_rows = n_train)
    elif idx_y == 1:
        y = np.genfromtxt(data_path + "{}_labels_{}{}_Im.txt".format(C_str, size_str, ext_str), delimiter = ',', max_rows = n_train)
    x = x[:,:]
    y = y[:,[0, 3]]
    #idcs = np.where(np.logical_and(np.abs(y[:,idx_y]) < 200, np.abs(y[:,idx_y]) > 1e-8))
    #idcs = np.where(np.logical_and(x[:,0] < 0, x[:,1] < 0.5))
    #x = x[idcs]
    #y = y[idcs]
    #x_test = np.random.shuffle(x_test)
    print(x.shape)

    return x, y


def get_real_part(y):
    return y[:,0]
def get_imag_part(y):
    return y[:,1]
######### Model Init ##################

def init_model(ndims, nout, nodes, layers, activation, loss):

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
    return model

def forward_pass(model, x_test, scaler):
    start = time.time()
    y_pred = np.array(model(x_test))
    end = time.time() - start
    print("Time to execute forward pass = ", end, " seconds")

    #y_pred = np.float64(y_pred)
    y_pred = scaler.inverse_transform(y_pred).squeeze()
    return y_pred

def get_measures(y_test, y_pred):
    diff = y_test - y_pred
    errRe = np.divide(diff, y_test, out=np.zeros_like(diff), where=y_test!=0)
    #errRe = np.where(np.logical_or(y_test > 0., y_test < 0.), np.array((y_test - y_pred)/y_test), 0)
    args = np.argsort(np.abs(errRe))
    idcs = (-np.abs(errRe)).argsort()[:500]
    #print(y_test[idcs, 0])
    #print(y_pred[idcs])
    mean_err = np.mean(np.abs(errRe))*100
    median_err = np.median(np.abs(errRe))*100
    pts95_1 = (np.logical_and(errRe < 0.01, errRe > -0.01)).sum()
    pts95_5 = (np.logical_and(errRe < 0.05, errRe > -0.05)).sum()
    pts95_10 = (np.logical_and(errRe < 0.1, errRe > -0.1)).sum()
    eta_1 = pts95_1/errRe.size * 100
    eta_5 = pts95_5/errRe.size * 100
    eta_10 = pts95_10/errRe.size * 100
    return errRe, args, idcs, mean_err, median_err, pts95_1, pts95_5, pts95_10, eta_1, eta_5, eta_10

def plot_everything(C_str, NAME, mean_err, median_err, eta_1, eta_5, eta_10, history, y_test, y_pred, idcs, errRe, x_test, logerr):

    plot.plot_err(NAME, errRe, mean_err, median_err, eta_1, eta_5, eta_10)
    plot.plot_err_feat(C_str, NAME, errRe, x_test)
    plot.plot_loss(NAME, history)
    #plot_worst(NAME, y_test, y_pred, idcs)

    plot.plot_logerr(NAME, logerr)
    plot.plot_all(NAME, mean_err, median_err, eta_1, eta_5, eta_10, history, y_test, y_pred, idcs, errRe, x_test, logerr)
def execute(data_path, C_str_feat, C_str, size_str, size_str_test, ext_str ,norm, loss, idx, idx_y,
            ndims, nout, nodes, layers, activation, epochs, batch, load):

    data, labels = get_data(data_path, C_str_feat, C_str, size_str, ext_str, idx, idx_y)
    x_test, y_test = get_test_data(data_path, C_str_feat, C_str, size_str_test,ext_str_test, idx, idx_y)

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
    while os.path.isfile(path + 'models/{}_{}.h5'.format(NAME, count)):
        count += 1
        if count > 100:
            break

    #x_test, y_test = get_test_data(path + 'ti3r1' +'_data/', 'ti3r1','C010mimi0', size_str_test,ext_str_test, idx)


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
    	y_train = scaler.fit_transform(y_train.reshape(-1,1))
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
        model = tf.keras.models.load_model(path + "models/{}.h5".format(NAME), compile = False)
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

        model = init_model(ndims, nout, nodes, layers, activation, loss)
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



    y_pred = forward_pass(model, x_test, scaler)
    print(y_pred)
    abserr = np.abs(y_test - y_pred)
    errRe, args, idcs, mean_err, median_err,pts95_1, pts95_5, pts95_10, eta_1, eta_5, eta_10 = get_measures(y_test, y_pred)
    ratio = np.divide(y_pred, y_test, out=np.ones_like(y_pred), where=np.logical_and(y_test!=0, y_pred!=0))
    #print(ratio[ratio<1e-7])
    logerr = np.log(np.abs(ratio))
    #plot.plot_2D(C_str, NAME, abserr, x_test, 'Absolute Error', r'$\frac{t}{s}$', r'$\frac{m^2}{s}$', 'abserr', 'log')
    #plot.plot_2D(C_str, NAME, y_pred, x_test, 'Neural Network', r'$\frac{t}{s}$', r'$\frac{m^2}{s}$', 'NN', 'symlog')
    #plot.plot_2D(C_str, NAME, y_test, x_test, 'Actual', r'$\frac{t}{s}$', r'$\frac{m^2}{s}$', 'actual', 'symlog')
    plot_everything(C_str, NAME, mean_err, median_err, eta_1, eta_5, eta_10, history, y_test, y_pred, idcs, errRe, x_test, logerr)

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

    if load == 0:
        load = False
    elif load == 1:
        load = True

    N_func = 4
    func_string=['ti3r1', 'ti3r1', 'ti3r1']

    """
    for i in range(N_func):
        for j in range(2):
            execute(data_path, C_str_feat, func_string[i], size_str, size_str_test, ext_str, norm, loss, i, j,
                    ndims, nodes, layers, activation, epochs, batch, load))
    """

    execute(data_path, func_string[0], func_string[0], size_str, size_str_test, ext_str, norm, loss, 0, 0,
            ndims,nout, nodes, layers, activation, epochs, batch, load)

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
