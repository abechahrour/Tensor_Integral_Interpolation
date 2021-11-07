import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
from matplotlib import gridspec
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback, CSVLogger
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K
import matplotlib as mpl
#from kerastuner.tuners import RandomSearch
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from scipy.stats import binned_statistic_2d
tf.keras.backend.set_floatx('float64')

parent = 'ti3r1'
path = '/home/chahrour/golem_NN/' + parent + '/'

"""
x_bins = np.linspace(0, 1, 100)
y_bins = np.linspace(0, 1, 100)
xmin = 1e-5
xmax = 1.
ymin = 1e-5
ymax = 1.
x = np.random.rand(1000)
y = np.random.rand(1000)
z = x+y
ret = binned_statistic_2d(x, y, z, statistic=np.mean, bins = 10, range=[[xmin,xmax],[ymin,ymax]])
plt.imshow(ret.statistic.T, origin='bottom')
plt.colorbar()
#plt.xscale('symlog', linthreshx=1e-5)
#plt.yscale('symlog', linthreshy=1e-5)
plt.savefig("test.png")
print(np.shape(ret))
print(ret)
"""


def plot_err(NAME, errRe, mean_err, median_err, eta_1, eta_5, eta_10):

    plt.figure()
    plt.hist(100*errRe, bins = 1000, histtype = 'step')
    plt.yscale('log')
    plt.xlabel('Relative error (%)')
    plt.ylabel('Counts')
    plt.suptitle('Relative Error Distribution')
    plt.title(NAME)
    plt.savefig(path + "plots/errors/errors_{}.png".format(NAME), bbox_inches="tight")

    plt.figure()
    plt.hist(100*errRe, bins = 100, histtype = 'step', range = (-100,100))
    plt.yscale('log')
    plt.xlabel('Relative error (%)')
    plt.ylabel('Counts')
    plt.suptitle('Relative Error Distribution')
    plt.title(NAME)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax = plt.gca()
    ax.text(0.3, 0.90, r'$\epsilon$ = {:0.3f} %'.format(mean_err),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
    ax.text(0.3, 0.80, r'$\epsilon_m$ = {:0.3f} %'.format(median_err),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
    ax.text(0.3, 0.70, r'$\eta1$ = {:0.3f} %'.format(eta_1),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
    ax.text(0.3, 0.60, r'$\eta5$ = {:0.3f} %'.format(eta_5),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
    ax.text(0.3, 0.50, r'$\eta10$ = {:0.3f} %'.format(eta_10),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
    plt.savefig(path + "plots/errors_zoom/errors_zoom_{}.png".format(NAME), bbox_inches="tight")


def plot_loss(NAME, history):


    fig = plt.figure()
    # set height ratios for sublots
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    # the fisrt subplot
    ax0 = plt.subplot(gs[0])
    # log scale for axis Y of the first subplot
    ax0.set_yscale("log")
    ax0.set_ylabel("Loss")
    ax0.set_xlabel("Epochs")
    #ax0.set_suptitle('Training History')
    ax0.set_title("Training History \n" + NAME)
    #ax0.plot(x, y, color='r')
    ax0.plot(history.history['loss'], label='Training loss')
    ax0.plot(history.history['val_loss'], label = 'Validation Loss', alpha=0.4)
    #the second subplot
    # shared axis X
    ax1 = plt.subplot(gs[1], sharex = ax0)
    ax1.plot(history.history['lr'], label = 'Learning Rate', color='b')
    plt.setp(ax0.get_xticklabels(), visible=False)
    # remove last tick label for the second subplot
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

    # put lened on first subplot
    ax0.legend()

    # remove vertical gap between subplots
    plt.subplots_adjust(hspace=.0)
    plt.savefig(path + "plots/losses/loss_{}.png".format(NAME), bbox_inches="tight")
"""
def plot_worst(NAME, y_test, y_pred, idcs):
    plt.figure()
    plt.subplot(221)
    plt.hist(np.abs(y_test[idcs, 0]), bins = 50, histtype = 'step')
    plt.xscale('log')
    plt.xlabel('Actual Values')
    plt.ylabel('Counts')
    plt.subplot(222)
    plt.hist(np.abs(y_pred[idcs]), bins = 50, histtype = 'step')
    plt.xscale('log')
    plt.xlabel('Predicted Values')
    #plt.ylabel('Counts')
    #plt.tight_layout()
    #plt.savefig(path + "plots/preds/worstlog_{}.png".format(NAME), bbox_inches="tight")

    plt.subplot(223)
    plt.hist(np.abs(y_test[idcs, 0]), bins = 50, histtype = 'step')
    #plt.xscale('log')
    plt.xlabel('Actual Values')
    plt.ylabel('Counts')
    plt.subplot(224)
    plt.hist(np.abs(y_pred[idcs]), bins = 50, histtype = 'step')
    #plt.xscale('log')
    plt.xlabel('Predicted Values')
    #plt.ylabel('Counts')
    #plt.tight_layout()
    plt.savefig(path + "plots/preds/worst_{}.png".format(NAME), bbox_inches="tight")
"""

def plot_2D(C_str, NAME, vals, x_test, title, xlabel, ylabel, lead_str, norm = 'symlog'):

    if norm=='log':
        norm = mpl.colors.LogNorm()
    elif norm=='symlog':
        norm = mpl.colors.SymLogNorm(linthresh = 1e0)
    x_bins = np.linspace(0, 1, 100)
    y_bins = np.linspace(0, 1, 100)
    step = 1/70
    bins = 80
    #ret = binned_statistic_2d(x_test[:,0], x_test[:,1], np.abs(errRe*100), statistic=np.mean, bins =[bins, bins])
    H, xedges, yedges, binnumber = binned_statistic_2d(x_test[:,0], x_test[:,1], values = vals, statistic=np.mean , bins = [bins, bins])
    #H2, xedges2, yedges2, binnumber2 = stats.binned_statistic_2d(x, y, values = z, statistic='mean' , bins = [20, 20])

    XX, YY = np.meshgrid(xedges, yedges)
    #XX2, YY2 = np.meshgrid(xedges2, yedges2)
    plt.figure()
    plt.pcolormesh(XX,YY,H.T, norm= norm, cmap='plasma')
    #plt.imshow(ret.statistic.T,norm= mpl.colors.LogNorm(), origin='bottom', cmap='plasma')
    #plt.hist2d(x_test[:,0], x_test[:,1], bins = bins, weights = ret.statistic.T, norm= mpl.colors.LogNorm(), cmap=plt.cm.jet)
    plt.colorbar()
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    #plt.ylabel('Counts')
    #plt.tight_layout()
    #plt.ylim(-10, 25)
    #plt.suptitle(r'Relative Error vs. $r$')
    plt.title("{} {}".format(C_str, title))
    #plt.tight_layout()
    plt.savefig(path + "plots/preds/{}_{}.png".format(lead_str,NAME), bbox_inches="tight")

def plot_err_feat(C_str, NAME, errRe, x_test):
    x_bins = np.linspace(0, 1, 100)
    y_bins = np.linspace(0, 1, 100)
    step = 1/70
    bins = 80
    H, xedges, yedges, binnumber = binned_statistic_2d(x_test[:,0], x_test[:,1], values = np.abs(errRe)*100, statistic=np.mean , bins = [bins, bins])
    #H2, xedges2, yedges2, binnumber2 = stats.binned_statistic_2d(x, y, values = z, statistic='mean' , bins = [20, 20])

    XX, YY = np.meshgrid(xedges, yedges)
    #XX2, YY2 = np.meshgrid(xedges2, yedges2)
    plt.figure()
    plt.pcolormesh(XX,YY,H.T, norm= mpl.colors.LogNorm(), cmap='plasma')
    #plt.imshow(ret.statistic.T,norm= mpl.colors.LogNorm(), origin='bottom', cmap='plasma')
    #plt.hist2d(x_test[:,0], x_test[:,1], bins = bins, weights = ret.statistic.T, norm= mpl.colors.LogNorm(), cmap=plt.cm.jet)
    plt.colorbar()
    plt.ylabel(r'$\frac{m^2}{s}$', fontsize=20)
    plt.xlabel(r'$\frac{t}{s}$', fontsize=20)
    #plt.ylabel('Counts')
    #plt.tight_layout()
    #plt.ylim(-10, 25)
    #plt.suptitle(r'Relative Error vs. $r$')
    plt.title("{} Relative Error (%)".format(C_str))
    #plt.tight_layout()
    plt.savefig(path + "plots/preds/errfeat_{}.png".format(NAME), bbox_inches="tight")

def plot_logerr(NAME, err):
    plt.figure()
    plt.hist(err, bins = 1000, histtype = 'step')
    plt.yscale('log')
    plt.xlabel('Log-Accuracy')
    plt.ylabel('Counts')
    plt.suptitle('Log-Accuracy Distribution')
    plt.title(NAME)
    plt.savefig(path + "plots/errors/logacc_{}.png".format(NAME), bbox_inches="tight")

def plot_all(NAME, mean_err, median_err, eta_1, eta_5, eta_10, history, y_test, y_pred, idcs, errRe, x_test, err):
    plt.figure(figsize=(16,9))
    plt.suptitle(NAME);
    plt.subplot(221)
    plt.hist(100*errRe, bins = 1000, histtype = 'step')
    plt.yscale('log')
    plt.xlabel('Relative error (%)')
    plt.ylabel('Counts')
    #plt.suptitle('Relative Error Distribution')

    plt.subplot(222)
    plt.hist(100*errRe, bins = 100, histtype = 'step', range = (-100,100))
    plt.yscale('log')
    plt.xlabel('Relative error (%)')
    plt.ylabel('Counts')
    #plt.suptitle('Relative Error Distribution')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax = plt.gca()
    ax.text(0.3, 0.90, r'$\epsilon$ = {:0.3f} %'.format(mean_err),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
    ax.text(0.3, 0.80, r'$\epsilon_m$ = {:0.3f} %'.format(median_err),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
    ax.text(0.3, 0.70, r'$\eta1$ = {:0.3f} %'.format(eta_1),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
    ax.text(0.3, 0.60, r'$\eta5$ = {:0.3f} %'.format(eta_5),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
    ax.text(0.3, 0.50, r'$\eta10$ = {:0.3f} %'.format(eta_10),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
    plt.subplot(223)
    plt.hist(err, bins = 1000, histtype = 'step')
    plt.yscale('log')
    plt.xlabel('Log-Accuracy')
    plt.ylabel('Counts')
    #plt.suptitle('Log-Accuracy Distribution')

    plt.subplot(224)
    step = 1/70
    bins =100
    ret = binned_statistic_2d(x_test[:,0], x_test[:,1], np.abs(errRe*100), statistic=np.mean, bins =[np.arange(0,1+step,step), np.arange(0,1+step,step)])
    plt.figure()
    plt.imshow(ret.statistic.T,norm= mpl.colors.LogNorm(), origin='bottom', cmap='plasma')
    #plt.hist2d(x_test[:,0], x_test[:,1], bins = bins, weights = ret.statistic.T, norm= mpl.colors.LogNorm(), cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel(r'r ($\frac{m_0^2}{q^2}$)')
    plt.ylabel(r's ($\frac{m_1^2}{q^2}$)')
    #plt.ylabel('Counts')
    #plt.tight_layout()
    #plt.ylim(-10, 25)
    #plt.suptitle(r'Relative Error vs. $r$')


    plt.tight_layout(pad=2)
    plt.savefig(path + "plots/all/all_{}.png".format(NAME), bbox_inches="tight")
