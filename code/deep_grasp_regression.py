# -*- coding: utf-8 -*-
"""Simultaneous and proportional control using regression-CNN.

Data from MeganePro dataset 1
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1Z3IOM

CNN architecture adapted from
Ameri, Ali, et al. "Regression convolutional neural network for improved simultaneous
EMG control." Journal of neural engineering 16.3 (2019): 036015.

example call:
python deepGraspRegression.py
--datapath ..\data\S010_ex1.mat
--scalex no
--algo cnn
--window 400
--stride 20
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import lfilter, butter
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, \
    mean_squared_error
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy import signal
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import time

import myutils_tensorflow as tf_utils


TENSORFLOWDIRNAME = "toy_windowing_CNN_{}".format(int(time.time()))


# region functions hudgins features


def extract_hudgins_features(emg, w_dim=400, stride=20):
    """
    Extraction of hudgins features on moving window.
    """

    assert len(emg.shape) <= 2
    if len(emg.shape) == 1:
        x = emg[:, None]
    else:
        x = emg
    diff_x = np.diff(x, axis=0)  # to avoid re-calculations

    # mav
    mav = moving_function(np.abs(x), "mean", w_dim, stride)

    # num zero crossings
    zc = np.vstack((np.zeros(x.shape[1]),
                    np.diff(x >= 0, axis=0)))
    n_zc = moving_function(zc, "sum", w_dim, stride)

    # num slope sign changes
    t = 0  # 0.000001 threshold may be determined observing f,b = np.histogram(np.abs(
    # diff_x.ravel()),bins=1000)
    ssc = np.logical_and(
        np.vstack((np.zeros(x.shape[1]), np.diff(diff_x >= 0, axis=0) != 0,
                   np.zeros(x.shape[1]))),
        np.logical_or(np.vstack((np.abs(diff_x) >= t, np.zeros(x.shape[1]))),
                      # samples followed by a big jump
                      np.vstack((np.zeros(x.shape[1]),
                                 np.flipud(np.abs(np.diff(np.flipud(x), axis=0)) >= t))))
        # samples preceeded by a big jump
    )
    n_ssc = moving_function(ssc, "sum", w_dim, stride)

    # waveform length
    wl = moving_function(np.abs(diff_x), "behaded_sum", w_dim, stride)

    # put all together mav0, nzc0, nssc0, wl0, mav1, nzc1, ...
    n_feats = 4  # number of features to be computed
    n_w = 1 + np.floor_divide(x.shape[0] - w_dim,
                              stride)  # number of resulting windows, i.e. samples
    n_feats_tot = x.shape[1] * n_feats
    emg = np.empty((n_w, n_feats_tot))
    for i in range(x.shape[1]):
        emg[:, i * 4] = mav[:, i]
        emg[:, i * 4 + 1] = n_zc[:, i]
        emg[:, i * 4 + 2] = n_ssc[:, i]
        emg[:, i * 4 + 3] = wl[:, i]

    return emg


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y


# endregion


# region functions megane pro


def load_megane_database(fname, variable_names, train_reps, test_reps, subsamplerate=1,
                         feature_set=[], w_len=400, w_stride=20):
    """ Load the desired fields (variable_names) from the megane pro database
    """

    data = loadmat(fname, variable_names=variable_names[:])

    # preprocessing
    data["emg"] = preprocess(data["emg"], feature_set=feature_set)

    # feature extraction
    if "raw" in feature_set:  # do nothing
        pass
    if "im" in feature_set:  # do nothing
        pass
    if "hudgins" in feature_set:
        data["emg"] = extract_hudgins_features(data["emg"], w_dim=w_len, stride=w_stride)
        data["regrasp"] = subsample_data_window_based(
            data["regrasp"], w_dim=w_len, stride=w_stride)
        data["regrasprepetition"] = subsample_data_window_based(
            data["regrasprepetition"], w_dim=w_len, stride=w_stride)
        data["ts"] = subsample_data_window_based(
            data["ts"], w_dim=w_len, stride=w_stride)
        data["reobject"] = subsample_data_window_based(
            data["reobject"], w_dim=w_len, stride=w_stride)
        data["reposition"] = subsample_data_window_based(
            data["reposition"], w_dim=w_len, stride=w_stride)
        data["redynamic"] = subsample_data_window_based(
            data["redynamic"], w_dim=w_len, stride=w_stride)

    data["regrasp"] = convert_label(data["regrasp"])

    # split database
    x_train, y_train, x_test, y_test, cv_splits = split_data(data, train_reps,
                                                             test_reps,
                                                             debug_plot=False,
                                                             subsamplerate=subsamplerate)
    return x_train, y_train, x_test, y_test, cv_splits


def preprocess(emg, feature_set=[]):
    """
    if extract_features is [] then:
    * rectify and filter emg data.
    * 2nd order butterworth filter with cut-off 10Hz

    if extract_features contains ["hudgins"]:
    * 2nd order butterworth band-pass filter with cut-off 10Hz
    """

    if "im" in feature_set:
        emg = np.abs(emg)  # rectification
        sos = butter(2, 10, "lp", fs=1926, output="sos")
        emg = signal.sosfilt(sos, emg, axis=0)  # low-pass 10 Hz
    else:
        if "hudgins" in feature_set:
            emg = butter_bandpass_filter(emg, lowcut=10, highcut=500, fs=1926,
                                         order=2)  # band-pass 10-500 Hz

    return emg


def moving_function(data, func, w_dim, stride=1):
    """
    Optimized computation of cumulative functions over a moving window.
    Data can be 1 or 2 dimensional array.
    Options for the function are "sum", "mean", "beheaded_sum".
    The "beheaded_sum" sums all the elements except the last in the window.
    """

    assert len(data.shape) <= 2
    if len(data.shape) == 1:  # if 1d array -> make column array
        one_dim_input = True
        data = data[:, None]
    else:
        one_dim_input = False

    if func == "mean":
        coeff = 1 / w_dim
        mask = np.ones((w_dim,))
    if func == "sum":
        coeff = 1
        mask = np.ones((w_dim,))
    if func == "behaded_sum":
        coeff = 1
        mask = np.concatenate((np.ones((w_dim - 1,)), [0]))

    tmp = []
    for c in data.T:
        tmp.append(np.convolve(c, mask * coeff, mode="valid"))
    tmp = np.hstack([c[::stride, None] for c in tmp])

    if one_dim_input:
        tmp = tmp.ravel()

    return tmp


def subsample_data_window_based(x, w_dim, stride=1):
    """
    Subsample the labels in case window features have been extracted
    """

    idx = np.arange(w_dim - 1, x.shape[0], stride)
    if len(x.shape) == 1:
        x = x[idx]
    else:
        x = x[idx, :]
    return x


def convert_label(label):
    """
    Converts the grasp label (column (re)grasp) into a regression-like target
    value. Grasp types are converted into the 6dof configuration of the fingers.
    Multiple* grasp types are converted to a simple power grasp [1,0,1,1,1,1]..
    Original grasps:
    0, resting hand, [0,0,0,0,0,0]
    1, medium wrap*, [1,0,1,1,1,1]
    2, lateral, [1,1,1,1,1,1]
    3, parallel extension*, [1,0,1,1,1,1]
    4, tripod grasp, [1,0,1,1,0,0]
    5, power sphere*, [1,0,1,1,1,1]
    6, precision disk*, [1,0,1,1,1,1]
    7, prismatic pinch, [1,0,1,0,0,0]
    8, index finger extension, [1,0,0,1,1,1]
    9, adducted thumb, [0,1,1,1,1,1]
    10, prismatic four finger, [1,0,1,1,1,0]
    """

    regression_label = np.zeros((label.shape[0], 6))
    actions_dict = {0: [0, 0, 0, 0, 0, 0], 1: [1, 0, 1, 1, 1, 1], 2: [1, 1, 1, 1, 1, 1],
                    3: [1, 0, 1, 1, 1, 1], 4: [1, 0, 1, 1, 0, 0], 5: [1, 0, 1, 1, 1, 1],
                    6: [1, 0, 1, 1, 1, 1], 7: [1, 0, 1, 0, 0, 0], 8: [1, 0, 0, 1, 1, 1],
                    9: [0, 1, 1, 1, 1, 1], 10: [1, 0, 1, 1, 1, 0]}
    for i in range(len(label)):
        regression_label[i] = actions_dict.get(label[i][0])
    return regression_label


def split_data(data, train_reps, test_reps, debug_plot=False, subsamplerate=1):
    """
    Obtain training and test set from dataset based on the grasp repetition number.

    The function assumes that
    * each static grasp is repeated on 3 objects, 4 times, sitting and standing;
    * each dynamic grasp is repeated on 2 objects, 4 times, standing;
    * in the DB, each repetition is numbered univocally within the same grasp, for example:
    grasp1 (static)(video)-5,-4,-3,-2,-1,0,(sitting)1,2,3,4,5,6,7,8,9,10,11,12,
    (standing)1,2,3,4,5,6,7,8,9,10,11,12;
    grasp11 (dynamic)(video)-3,-2,-1,0,(standing)1,2,3,4,5,6,7,8.

    Therefore, if the desired training repetitions are the first, the second and the
    third (train_reps=[1,2,3]), then the script will train on
    data["regrasprepetition"] == [1,2,3, 4,5,6, 7,8,9] for the static grasps and on
    data["regrasprepetition"] == [1,2, 3,4, 5,6] for the dynamic grasps.
    """

    # adjust train and test repetitions to the DB numbering
    train_reps_stat = []
    train_reps_dyn = []
    test_reps_stat = []
    test_reps_dyn = []
    for i in train_reps:
        train_reps_stat.append(i * 3 - 2)  # static sitting tasks
        train_reps_stat.append(i * 3 - 1)
        train_reps_stat.append(i * 3)
    for i in train_reps:
        train_reps_stat.append(i * 3 + 12 - 2)  # static standing tasks
        train_reps_stat.append(i * 3 + 12 - 1)
        train_reps_stat.append(i * 3 + 12)
    for i in train_reps:
        train_reps_dyn.append(i * 2 - 1)  # functional tasks
        train_reps_dyn.append(i * 2)
    for i in test_reps:
        test_reps_stat.append(i * 3 - 2)  # static sitting tasks
        test_reps_stat.append(i * 3 - 1)
        test_reps_stat.append(i * 3)
    for i in test_reps:
        test_reps_stat.append(i * 3 + 12 - 2)  # static standing tasks
        test_reps_stat.append(i * 3 + 12 - 1)
        test_reps_stat.append(i * 3 + 12)
    for i in test_reps:
        test_reps_dyn.append(i * 2 - 1)  # functional tasks
        test_reps_dyn.append(i * 2)

    # compute sample indices of training and testing reps, and split data
    idx_train = np.where(np.logical_or(np.logical_and(data["redynamic"] == 0,
                                                      np.isin(data["regrasprepetition"],
                                                              train_reps_stat)),
                                       np.logical_and(data["redynamic"] == 1,
                                                      np.isin(data["regrasprepetition"],
                                                              train_reps_dyn))))[0]
    idx_test = np.where(np.logical_or(np.logical_and(data["redynamic"] == 0,
                                                     np.isin(data["regrasprepetition"],
                                                             test_reps_stat)),
                                      np.logical_and(data["redynamic"] == 1,
                                                     np.isin(data["regrasprepetition"],
                                                             test_reps_dyn))))[0]

    idx_train = idx_train[::subsamplerate]

    x_train = data["emg"][idx_train, :]
    y_train = data["regrasp"][idx_train, :]
    x_test = data["emg"][idx_test, :]
    y_test = data["regrasp"][idx_test, :]

    if debug_plot:
        plt.figure()
        plt.title("All grasp reps, train reps, test reps")
        plt.plot(data["regrasprepetition"]);
        plt.plot(idxmask2binmask(idx_train, 1));
        plt.plot(idxmask2binmask(idx_test, 1));

    # compute data splits for cross validation
    cv_splits = compute_cv_splits(data, idx_train, train_reps, debug_plot,
                                  subsample_cv=(subsamplerate!=1))

    return x_train, y_train, x_test, y_test, cv_splits


def compute_cv_splits(data, idx_train, train_reps, debug_plot, subsample_cv=False):
    """

    """

    reps_x_train = data["regrasprepetition"][idx_train, :]
    dyn_x_train = data["redynamic"][idx_train, :]

    # cv_test on one repetition and cv_train on the others
    cv_splits = []
    for i in train_reps:
        cv_test_rep_stat = []
        cv_test_rep_dyn = []
        cv_train_rep_stat = []
        cv_train_rep_dyn = []
        cv_test_rep_stat.append(i * 3 - 2)
        cv_test_rep_stat.append(i * 3 - 1)
        cv_test_rep_stat.append(i * 3)
        cv_test_rep_stat.append(i * 3 + 12 - 2)
        cv_test_rep_stat.append(i * 3 + 12 - 1)
        cv_test_rep_stat.append(i * 3 + 12)
        cv_test_rep_dyn.append(i * 2 - 1)
        cv_test_rep_dyn.append(i * 2)
        for j in train_reps[~(train_reps == i)]:
            cv_train_rep_stat.append(j * 3 - 2)
            cv_train_rep_stat.append(j * 3 - 1)
            cv_train_rep_stat.append(j * 3)
        for j in train_reps[~(train_reps == i)]:
            cv_train_rep_stat.append(j * 3 + 12 - 2)
            cv_train_rep_stat.append(j * 3 + 12 - 1)
            cv_train_rep_stat.append(j * 3 + 12)
        for j in train_reps[~(train_reps == i)]:
            cv_train_rep_dyn.append(j * 2 - 1)
            cv_train_rep_dyn.append(j * 2)

        cv_idx_train = np.logical_or(np.logical_and(dyn_x_train == 0,
                                                    np.isin(reps_x_train,
                                                            cv_train_rep_stat)),
                                     np.logical_and(dyn_x_train == 1,
                                                    np.isin(reps_x_train,
                                                            cv_train_rep_dyn))).ravel()
        cv_idx_test = np.logical_or(np.logical_and(dyn_x_train == 0,
                                                   np.isin(reps_x_train,
                                                           cv_test_rep_stat)),
                                    np.logical_and(dyn_x_train == 1,
                                                   np.isin(reps_x_train,
                                                           cv_test_rep_dyn))).ravel()

        cv_splits.append((np.where(cv_idx_train)[0], np.where(cv_idx_test)[0]))

    if subsample_cv:
        for i in range(len(cv_splits)):
            cv_splits[i] = (cv_splits[i][0][::4], cv_splits[i][1][::4])

    if debug_plot:
        plt.figure();
        plt.title("Train grasp reps, 3-fold cv splits (2train, 1test each split)")
        for i, (cv_idx_train, cv_idx_test) in enumerate(cv_splits):
            plt.plot(reps_x_train);
            plt.plot(idxmask2binmask(cv_idx_train, -i));
            plt.plot(idxmask2binmask(cv_idx_test, -i));

    # TODO: write CV visualization function (https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py)

    return cv_splits


def idxmask2binmask(m, coeff=1):
    """
    Converts an indices mask to a binary mask. The binary mask is == 1 in all
    the indices contained in the mask, and 0 otherwise.
    """
    m = np.array(m)
    assert len(m.shape) == 1 or (
            len(m.shape) == 2 and (m.shape[0] == 1 or m.shape[1] == 1))
    if len(m.shape) == 2:
        m = m.ravel()
    assert m[0] >= 0 and all(np.diff(m) > 0)
    v = np.zeros(np.max(m) + 1)
    v[m] = 1 * coeff
    return v


# endregion


# region functions misc


def generate_mock_binary_clf_database(n_samples, n_features):
    """ Generate a mock database for binary classification
    """

    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               random_state=1, n_classes=2)
    y = y[:, None]
    y = np.hstack((y, 1 - y))
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        shuffle=False)
    kf = KFold(n_splits=3)
    cv_splits = list(kf.split(x_train, y_train))
    return x_train, x_test, y_train, y_test, cv_splits


def quick_visualize_vec(data, overlay_data=None, title="", continue_on_fig=None, x_axis=None):
    '''
    Multiple line subplots, one for each column (up to the 12th column).
    If two matrices (data and overlay_data) are passed, the second will be overlayed
    to the first one.
    If an x_axis is passed, all the columns of the passed data will be plot against the
    common x_axis.
    If a figure handle is passed, the function will try to plot the data
    on that function (num_subplots must be equal to num columns!).
    '''

    if len(data.shape) > 2:
        print("Only 1 and 2d arrays are accepted")
        return None
    elif len(data.shape) == 1:
        data = data[:, None]
    if data.shape[1] > 12:
        print("Up to 12 columns can be printed!")
        return None
    if continue_on_fig and len(continue_on_fig.axes) != data.shape[1]:
        print("The number of data columns and of subplots passed must be the same!")
        return None

    if x_axis is not None:
        x_axis = x_axis.ravel()
        if len(x_axis) != data.shape[0]:
            print("x_axis must be a 1d array of the same size of data.shape[0]")
            return None

    if overlay_data is not None:
        if len(overlay_data.shape) > 2:
            print("Overlay_data not plotted. Only 1 and 2d arrays are accepted.")
        elif len(overlay_data.shape) == 1:
            overlay_data = overlay_data[:, None]
        if overlay_data.shape != data.shape:
            print("Overlay_data not plotted. It must have the same number of columns of "
                  "the main data.")
            return None

    n_subplot_cols = (data.shape[1] - 1) // 6 + 1
    n_subplot_rows = np.min([6, data.shape[1]])
    if not continue_on_fig:
        fig = plt.figure(figsize=figaspect(0.5))
        for i in range(1, data.shape[1] + 1):
            ax1 = fig.add_subplot(n_subplot_rows, n_subplot_cols, i)
            ax1.plot(data[:, i - 1]) if x_axis is None else ax1.plot(x_axis, data[:, i - 1])
            if overlay_data is not None:
                ax1.plot(overlay_data[:, i - 1]) if x_axis is None else ax1.plot(x_axis, overlay_data[:, i - 1])
    else:
        fig = continue_on_fig
        for i in range(data.shape[1]):
            fig.axes[i].plot(data[:, i]) if x_axis is None else fig.axes[i].plot(x_axis, data[:, i])
            if overlay_data is not None:
                fig.axes[i].plot(overlay_data[:, i]) if x_axis is None else fig.axes[i].plot(x_axis, overlay_data[:, i])
    plt.suptitle(title)
    return fig


# endregion


def main():

    # region parse parameters

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # dataset args
    parser.add_argument("--datapath", required=False, default="dummydb",
                        help="path to the database. If nothing is give, generate"
                             "dummy database. The path must be relative to the "
                             ".py script. Ex: path\\fname.mat.")
    parser.add_argument("--trainreps", required=False, default="1 2 3",
                        help="training repetitions (from 1 to 4). Ex: 1 2 3")
    parser.add_argument("--testreps", required=False, default="4",
                        help="test repetitions (from 1 to 4). Ex: 4")
    parser.add_argument("--alltraintestsplits", required=False, default=False, type=bool,
                        help="if True, overrides trainreps and testreps. Performance "
                             "will be computed over the average of all the possible "
                             "train-test splits 123-4,124-3,134-2,124-3. Default: False.")
    parser.add_argument("--subsamplerate", required=False, default=1, type=int,
                        help="how much to subsample the training set. A value v corresponds"
                             "to a subsampling of 1:v. Default: 1.")
    # features args
    parser.add_argument("--featureset", required=False, default="raw",
                        help='which feature set to use, for example raw, im (interactive '
                             'myocontrol), hudgins or a combination of features '
                             '"hudgins rms". '
                             'Default: "raw".')
    # transform x args
    parser.add_argument("--scalex", required=False, default="no",
                        choices=["yes", "no"],
                        help="whether to normalize the emg or not (yes/no). Default: no.")
    parser.add_argument("--transformx", required=False, default="none",
                        choices=["none", "rff", "rbf"],
                        help="how to transform the Xs (none, rff, rbf). Default: none.")
    parser.add_argument("--numRFFseeds", required=False, default=1, type=int,
                        help="on how many seeds to average the RFFs, in case they were "
                             "used. Default: 1.")
    parser.add_argument("--window", required=False, default=1, type=int,
                        help="dimension of sliding window. Default: 1.")
    parser.add_argument("--stride", required=False, default=1, type=int,
                        help="stride of the sliding window. Default: 1.")
    # transform y args
    parser.add_argument("--transformy", required=False, default="none",
                        choices=["none", "logit", "arctanh"],
                        help="how to transform the Ys (none, logit, arctanh). "
                             "Default: none.")
    # ml algo args
    parser.add_argument("--algo", required=True, default="rr",
                        choices=["rr", "logr", "cnn"],
                        help="which regression algorithm to use (rr, logr, cnn). Default: rr.")
    # hyp opt args
    parser.add_argument("--hypopt", required=False, default="no",
                        choices=["yes", "no"],
                        help="hyperparameter optimization (yes/no). Default: no.")
    parser.add_argument("--njobs", required=False, default=1, type=int,
                        help="number of jobs to use for cross validation. -1 to use all the "
                             "available jobs. Default: 1.")
    # store args
    parser.add_argument("--saveoutput", required=False,
                        help="if needed, path where to save the output file. Ex: "
                             ".\\results\\")
    args = parser.parse_args()


    # parse args
    ALLOWED_FEATURES = ["raw", "im", "hudgins"]

    if not args.alltraintestsplits:
        train_reps = np.fromstring(args.trainreps, dtype=np.int, sep=" ")
        test_reps = np.fromstring(args.testreps, dtype=np.int, sep=" ")
    else:
        # combination of possible train-test reps
        reps = (((1, 2, 3), (4)),
                ((1, 2, 4), (3)),
                ((1, 3, 4), (2)),
                ((2, 3, 4), (1)))

    if args.hypopt == "yes":
        alphas_vec = np.logspace(-4, 4, 5)
        gammas_vec = np.logspace(-4, 4, 5)
        Cs_vec = np.logspace(-4, 4, 5)
    else:
        alphas_vec = [1.0]
        gammas_vec = [0.5]
        Cs_vec = [1.0]

    feature_set = str.split(args.featureset, " ")

    w_len = args.window
    w_stride = args.stride

    assert all(item in ALLOWED_FEATURES for item in feature_set), "invalid feature_set"
    if "raw" in feature_set:
        feature_set = ["raw"]
    if "im" in feature_set:
        feature_set = ["im"]

    if args.transformy == "none":
        targetTransform = None

    db_suffix = os.path.basename(args.datapath).split(".")[0].replace("_", "") + "_"
    if "raw" in feature_set:
        feature_suffix = "raw" + "_"
    elif "im" in feature_set:
        feature_suffix = "im" + "_"
    else:
        feature_suffix = "".join(feature_set) + "_"
    if args.transformx == "none":
        xtransform_suffix = "notransfx" + "_"
    scaling_suffix = "scaledx" + "_" if args.scalex == "yes" else "unscaledx" + "_"
    if args.transformy == "none":
        ytransform_suffix = "notransfy" + "_"
    hypopt_suffix = "ho_" if args.hypopt == "yes" else "nho_"
    algo_suffix = args.algo + "_"

    if args.saveoutput:
        out_dirname = os.path.join(args.saveoutput, db_suffix.replace("_", "")) + "\\"
        out_fname = out_dirname + \
                    feature_suffix + scaling_suffix + xtransform_suffix + \
                    ytransform_suffix + hypopt_suffix + algo_suffix
        Path(out_dirname).mkdir(parents=True, exist_ok=True)
        sys.stdout = open(out_fname + "console.txt", "w")  # redirect console to file

    # endregion

    # region load and process data

    # load data
    if args.datapath == "dummydb":  # generate dummy database
        x_train, x_test, y_train, y_test, cv_splits = \
            generate_mock_binary_clf_database(n_samples=10000, n_features=12)
    else:  # load megane pro database
        fields = ["regrasp", "regrasprepetition", "reobject", "reposition", "redynamic",
                  "emg", "ts"]
        x_train, y_train, x_test, y_test, cv_splits = \
            load_megane_database(args.datapath, fields, train_reps, test_reps,
                                 args.subsamplerate, feature_set, w_len=1, w_stride=1)

    x_val = x_train[cv_splits[0][1]]
    y_val = y_train[cv_splits[0][1]]
    x_train = x_train[cv_splits[0][0]]
    y_train = y_train[cv_splits[0][0]]

    # convert to tensorflow database
    n_features = x_train.shape[1]
    n_targets = y_train.shape[1]

    args_window_fn = [x_train, y_train, w_len, w_stride]
    db_train = tf.data.Dataset.from_generator(
        generator=tf_utils.create_sliding_window_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([w_len, n_features]), n_targets),
        args=args_window_fn)
    db_train_expanded = db_train.map(lambda x, y: (tf.expand_dims(x, axis=-1), y))
    db_train_expanded_batched = db_train_expanded.batch(128)

    args_window_fn = [x_val, y_val, w_len, w_stride]
    db_val = tf.data.Dataset.from_generator(
        generator=tf_utils.create_sliding_window_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([w_len, n_features]), n_targets),
        args=args_window_fn)
    db_val_expanded = db_val.map(lambda x, y: (tf.expand_dims(x, axis=-1), y))
    db_val_expanded_batched = db_val_expanded.batch(128)

    args_window_fn = [x_test, y_test, w_len, w_stride]
    db_test = tf.data.Dataset.from_generator(
        generator=tf_utils.create_sliding_window_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([w_len, n_features]), n_targets),
        args=args_window_fn)
    db_test_expanded = db_test.map(lambda x, y: (tf.expand_dims(x, axis=-1), y))
    db_test_expanded_batched = db_test_expanded.batch(128)


    # endregion


    # region MLP

    

    # endregion


    # region CNN regression

    # define tensorboard callback
    tensorboard = TensorBoard(log_dir="logs\\{}".format(TENSORFLOWDIRNAME))

    # define model
    myo_cnn_in = tf.keras.Input(shape=(400, 12, 1), name="in")

    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same", name="cnn1")(myo_cnn_in)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="pool1")(x)

    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same", name="cnn2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="pool2")(x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="cnn3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="pool3")(x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="cnn4")(x)
    x = tf.keras.layers.BatchNormalization(name="bn4")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="pool4")(x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="cnn5")(x)
    x = tf.keras.layers.BatchNormalization(name="bn5")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="pool5")(x)

    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="cnn6")(x)
    x = tf.keras.layers.BatchNormalization(name="bn6")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="pool6")(x)

    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same", name="cnn7")(x)
    x = tf.keras.layers.BatchNormalization(name="bn7")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="pool7")(x)

    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same", name="cnn8")(x)
    x = tf.keras.layers.BatchNormalization(name="bn8")(x)

    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(6, activation="relu", name="FC1")(x)
    myo_cnn_out = tf.keras.layers.Dense(6, activation="linear", name="out")(x)

    myo_cnn = tf.keras.Model(myo_cnn_in, myo_cnn_out, name="myocnn")
    myo_cnn.summary()

    myo_cnn.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.RMSprop(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    history = myo_cnn.fit(
        x=db_train_expanded_batched,
        epochs=50,
        validation_data=db_val_expanded_batched,
        callbacks=[tensorboard]
    )

    tf_utils.plot_history(history, metrics=["mean_absolute_error"], plot_validation=True)

    y_pred = myo_cnn.predict(db_test_expanded_batched)

    # re-extract y_test from tf.Dataset because they were subsampled by the sliding window
    y_test = tf_utils.get_db_elems_labels(db_test_expanded_batched)
    test_mse = tf.reduce_mean(tf.keras.metrics.mean_absolute_error(y_test, y_pred)).numpy()
    fig = quick_visualize_vec(y_test, y_pred, title="y_test vs y_true, MSE = "+str(test_mse))

    plt.plot()
    # endregion


if __name__ == "__main__":
    main()
