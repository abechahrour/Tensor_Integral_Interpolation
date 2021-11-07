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
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
import TensIntFirstTerms as tift

lr = 1e-3
activation = 'elu'

def get_data_NFT(path_features, path_labels, n_train, low, high, x_low = 0, x_high = [999, 999]):

    print("Loading Data")
    x = np.array(pd.read_csv(path_features, delimiter=',', nrows = n_train, error_bad_lines=False))
    y = np.array(pd.read_csv(path_labels, delimiter = ',', nrows = n_train, error_bad_lines=False))
    print("Data Loaded")
    print("Shape of x: ", x.shape)
    idx = np.argwhere(np.all(y[..., :] == 0, axis=0))
    y = np.delete(y, idx, axis=1)
    idcs = np.where(np.logical_and(x[:,0] < x_high[0], x[:,1] < x_high[1]))
    x = x[idcs]
    y = y[idcs]
    if path_features[6] == '1':
        y = y - tift.k1FirstTerm(x[:,0], x[:,1])
    elif path_features[6] == '2':
        y = y - tift.k2FirstTerm(x[:,0], x[:,1])
    elif path_features[6] == '3':
        y = y - tift.k3FirstTerm(x[:,0], x[:,1])
    elif path_features[6] == '4':
        y = y - tift.k4FirstTerm(x[:,0], x[:,1])
    #y = y[:,[0, 2, 3]]
    idx = np.argwhere(np.any(np.logical_or(np.abs(y[...,:]) > high, np.abs(y[...,:]) < low), axis=1))
    x =  np.delete(x, idx, axis=0)
    y = np.delete(y, idx, axis=0)
    return x, y

def get_data_comp(path_features, path_labels, n_train, comp, low, high):

    print("Loading Data")
    x = np.array(pd.read_csv(path_features, delimiter=',', nrows = n_train, error_bad_lines=False))
    y = np.array(pd.read_csv(path_labels, delimiter = ',', nrows = n_train, error_bad_lines=False))
    print("Data Loaded")
    print("Shape of x: ", x.shape)
    idx = np.argwhere(np.all(y[..., :] == 0, axis=0))
    y = np.delete(y, idx, axis=1)
    y = y[:,comp]
    idx = np.argwhere(np.logical_or(np.abs(y) > high, np.abs(y) < low))
    x =  np.delete(x, idx, axis=0)
    y = np.delete(y, idx, axis=0)
    return x, y
def get_data_new(path_features, path_labels, n_train, low, high, x_low = 0, x_high = [999, 999]):

    print("Loading Data")
    x = np.array(pd.read_csv(path_features, delimiter=',', nrows = n_train))
    y = np.array(pd.read_csv(path_labels, delimiter = ',', nrows = n_train))
    x = x[~np.isnan(y).any(axis=1)]
    y = y[~np.isnan(y).any(axis=1)]
    print("Data Loaded")
    print("Shape of x: ", x.shape)
    idx = np.argwhere(np.all(y[..., :] == 0, axis=0))
    y = np.delete(y, idx, axis=1)
    idcs = np.where(np.logical_and(x[:,0] < x_high[0], x[:,1] < x_high[1]))
    x = x[idcs]
    y = y[idcs]
    #y = y[:,[0, 2, 3]]
    idx = np.argwhere(np.any(np.logical_or(np.abs(y[...,:]) > high, np.abs(y[...,:]) < low), axis=1))
    x =  np.delete(x, idx, axis=0)
    y = np.delete(y, idx, axis=0)

    return x, y

def get_data(data_path , C_str_feat, C_str, size_str, ext_str, idx, idx_y, n_train, low, high):

    print("Loading Data")
    x = np.array(pd.read_csv(data_path + "{}_features_{}{}.txt".format(C_str_feat, size_str, ext_str), delimiter=',', nrows = n_train))
    if idx_y==0:
        y = np.array(pd.read_csv(data_path + "{}_labels_{}{}_Re.txt".format(C_str, size_str, ext_str), delimiter = ',', nrows = n_train))
    elif idx_y == 1:
        y = np.array(pd.read_csv(data_path + "{}_labels_{}{}_Im.txt".format(C_str, size_str, ext_str), delimiter = ',', nrows = n_train))
    print("Data Loaded")
    print("Shape of x_train: ", x.shape)
    idx = np.argwhere(np.all(y[..., :] == 0, axis=0))
    y = np.delete(y, idx, axis=1)
    #y = y[:,[0, 2, 3]]
    idx = np.argwhere(np.any(np.logical_or(np.abs(y[...,:]) > high, np.abs(y[...,:]) < low), axis=1))
    x =  np.delete(x, idx, axis=0)
    y = np.delete(y, idx, axis=0)
    #x = x[np.logical_and.reduce((np.abs(y[:,0])<200, np.abs(y[:,0])>1e-5,np.abs(y[:,1])<high, np.abs(y[:,1])>1e-5, np.abs(y[:,2])<high, np.abs(y[:,2])>1e-5))]
    #y = y[np.logical_and.reduce((np.abs(y[:,0])<high, np.abs(y[:,0])>1e-5,np.abs(y[:,1])<200, np.abs(y[:,1])>1e-5, np.abs(y[:,2])<200, np.abs(y[:,2])>1e-5))]
    #print("y[:,2] < 1e-5", y[np.abs(y[:,2])<1e-5])
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

def get_test_data(data_path , C_str_feat, C_str, size_str, ext_str, idx, idx_y, n_test, low, high):
    x = np.array(pd.read_csv(data_path + "{}_features_{}{}.txt".format(C_str_feat, size_str, ext_str), delimiter=',', nrows = n_test))
    if idx_y==0:
        y = np.array(pd.read_csv(data_path + "{}_labels_{}{}_Re.txt".format(C_str, size_str, ext_str), delimiter = ',', nrows = n_test))
    elif idx_y == 1:
        y = np.array(pd.read_csv(data_path + "{}_labels_{}{}_Im.txt".format(C_str, size_str, ext_str), delimiter = ',', nrows = n_test))
    idx = np.argwhere(np.all(y[..., :] == 0, axis=0))
    y = np.delete(y, idx, axis=1)
    #y = y[:,[0, 2, 3]]
    idx = np.argwhere(np.any(np.logical_or(np.abs(y[...,:]) > high, np.abs(y[...,:]) < low), axis=1))
    x =  np.delete(x, idx, axis=0)
    y = np.delete(y, idx, axis=0)
    #idcs = np.where(np.logical_and(np.abs(y[:,idx_y]) < high, np.abs(y[:,idx_y]) > 1e-8))
    #idcs = np.where(np.logical_and(x[:,0] < 0, x[:,1] < 0.5))
    #x = x[idcs]
    #y = y[idcs]
    #x = np.random.shuffle(x)
    print(x.shape)

    return x, y


def get_real_part(y):
    return y[:,0]
def get_imag_part(y):
    return y[:,1]


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


def init_disconnected_model(ndims, nout, nodes, layers, activation, loss):

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

    merged = tf.keras.layers.concatenate([arr for arr in y])
    model = Model(inputs=x,outputs=merged)

    model.summary()
    print("The Learning Rate is  ", lr)
    opt = Adam(lr)
    model.compile(optimizer = opt,
                  loss = loss)
    return model

def forward_pass(model, x, scaler):
    start = time.time()
    y_pred = np.array(model(x))
    end = time.time() - start
    print("Time to execute forward pass = ", end, " seconds")

    print(y_pred.shape)
    y_pred = y_pred.squeeze()
    if y_pred.ndim == 1:
        y_pred = scaler.inverse_transform(y_pred.reshape(1, -1)).squeeze()
    else:
        y_pred = scaler.inverse_transform(y_pred).squeeze()
    return y_pred

def get_measures(y, y_pred):
    diff = y - y_pred
    errRe = np.divide(diff, (y+y_pred)/2., out=np.zeros_like(diff), where=y!=0)
    #errRe = np.where(np.logical_or(y > 0., y < 0.), np.array((y - y_pred)/y), 0)
    args = np.argsort(np.abs(errRe))
    idcs = (-np.abs(errRe)).argsort()[:500]
    #print(y[idcs, 0])
    #print(y_pred[idcs])
    mean_err = np.mean(np.abs(errRe))*100
    means = np.mean(np.abs(errRe), axis = 0)*100
    median_err = np.median(np.abs(errRe))*100
    pts95_1 = (np.logical_and(errRe < 0.01, errRe > -0.01)).sum()
    pts95_5 = (np.logical_and(errRe < 0.05, errRe > -0.05)).sum()
    pts95_10 = (np.logical_and(errRe < 0.1, errRe > -0.1)).sum()
    eta_1 = pts95_1/errRe.size * 100
    eta_5 = pts95_5/errRe.size * 100
    eta_10 = pts95_10/errRe.size * 100
    return errRe, args, idcs, means, mean_err, median_err, pts95_1, pts95_5, pts95_10, eta_1, eta_5, eta_10

def plot_everything(path, C_str, NAME, mean_err, median_err, eta_1, eta_5, eta_10, history, y, y_pred, idcs, errRe, x, logerr):

    print("shape of err = ", errRe.shape)
    plot.plot_err(path, NAME, errRe, mean_err, median_err, eta_1, eta_5, eta_10)
    #plot.plot_err_feat(C_str, NAME, errRe[:,2], x)
    plot.plot_loss(path, NAME, history)
    #plot_worst(NAME, y, y_pred, idcs)

    plot.plot_logerr(path, NAME, logerr)
    #plot.plot_all(path, NAME, mean_err, median_err, eta_1, eta_5, eta_10, history, y, y_pred, idcs, errRe, x, logerr)

def scale(norm, path, NAME, low = -1, high = 1):
    scaler_data = FunctionTransformer()

    if norm == 'mm':
    	scaler = MinMaxScaler((low, high))
    	#y_train = scaler.fit_transform(y_train)
    elif norm == 'ss':
    	scaler = StandardScaler()
    	#y_train = scaler.fit_transform(y_train)
    elif norm == 'log':
    	scaler = FunctionTransformer(np.log1p, inverse_func = np.expm1)
    	#y_train = scaler.transform(y_train + np.abs(ymin))
    else:
        scaler = FunctionTransformer()



    return scaler_data, scaler
