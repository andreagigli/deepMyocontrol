# -*- coding: utf-8 -*-

import argparse
import os
import ntpath
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sys
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, \
    mean_squared_error, make_scorer


#############################################################
#################### ACCESSORY FUNCTIONS ####################
#############################################################

# TODO: delete this
# def import_data_matrix(datapath):
#     """
#     Imports the fields of interest from the database.
#     The fields are described in cognolato20gaze, table 4.
#     Fields of interest and zero based original_idx:
#     * (re)grasp (7)
#     * (re)grasprepetition (8)
#     * (re)object (9)
#     * (re)position (12)
#     * (re)dynamic (13)
#     * emg (50-61)
#     * ts (103)
#     """
#
#     fields = ['regrasp', 'regrasprepetition', 'reobject', 'reposition', 'redynamic',
#               'emg', 'ts']
#
#     d = loadmat(datapath, variable_names=fields)
#     data = np.array(d.get(fields[0]))
#     for key in fields[1:]:
#         data = np.hstack((data, d.get(key)))
#
#     return data


def convert_label(label):
    '''
    Converts the grasp label (column (re)grasp) into a regression-like target
    value. Grasp types are converted into the 6dof configuration of the fingers.
    Multiple* grasp types are converted to a simple power grasp [1,0,1,1,1,1]..
    Original grasps:
    0, resting hand, [0,0,0,0,0,0]
    1, medium wrap*, [1,0,1,1,1,1]
    2, lateral, [1,1,1,1,1,1]
    3, parallel extension*, [1,0,1,1,1,1]
    4, tripod grasp*, [1,0,1,1,0,0]
    5, power sphere*, [1,0,1,1,1,1]
    6, precision disk*, [1,0,1,1,1,1]
    7, prismatic pinch, [1,0,1,0,0,0]
    8, index finger extension, [1,0,0,1,1,1]
    9, adducted thumb, [0,1,1,1,1,1]
    10, prismatic four finger, [1,0,1,1,1,0]
    '''

    regression_label = np.zeros((label.shape[0], 6))
    actions_dict = {0: [0, 0, 0, 0, 0, 0], 1: [1, 0, 1, 1, 1, 1], 2: [1, 1, 1, 1, 1, 1],
                    3: [1, 0, 1, 1, 1, 1], 4: [1, 0, 1, 1, 0, 0], 5: [1, 0, 1, 1, 1, 1],
                    6: [1, 0, 1, 1, 1, 1], 7: [1, 0, 1, 0, 0, 0], 8: [1, 0, 0, 1, 1, 1],
                    9: [0, 1, 1, 1, 1, 1], 10: [1, 0, 1, 1, 1, 0]}
    for i in range(len(label)):
        regression_label[i] = actions_dict.get(label[i][0])
    return regression_label


def extract_features(data):
    """
    placeholder
    """

    return data


def split_data(data, train_reps, test_reps, debug_plot=False, subsample_train=False):
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

    if subsample_train == True:
        # TODO: implement subsampling and compute cv_splits accordingly (training data
        #  1:10, validation data further 1:4 over the training data)
        idx_train = idx_train[::10]

    x_train = data["emg"][idx_train, :]
    y_train = data["regrasp"][idx_train, :]
    x_test = data["emg"][idx_test, :]
    y_test = data["regrasp"][idx_test, :]

    if debug_plot:
        plt.figure()
        plt.title('All grasp reps, train reps, test reps')
        plt.plot(data["regrasprepetition"]);
        plt.plot(idxmask2binmask(idx_train,1));
        plt.plot(idxmask2binmask(idx_test,1));

    # compute data splits for cross validation
    cv_splits = compute_cv_splits(data, idx_train, train_reps, debug_plot,
                                  subsample_train)

    return x_train, y_train, x_test, y_test, cv_splits


def compute_cv_splits(data, idx_train, train_reps, debug_plot, subsample_val=False):
    '''

    '''

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

    if debug_plot:
        plt.figure();
        plt.title('Train grasp reps, 3-fold cv splits (2train, 1test each split)')
        for i, (cv_idx_train, cv_idx_test) in enumerate(cv_splits):
            plt.plot(reps_x_train);
            plt.plot(idxmask2binmask(cv_idx_train, -i));
            plt.plot(idxmask2binmask(cv_idx_test, -i));

    return cv_splits


def multiple_log_reg(x_train, y_train, x_test, y_test, cv_splits, logr_pipe,
                     logr_param_grid_pipe):
    '''
    Train logistic regression separately on each dof
    '''

    logr_y_pred = np.zeros(y_test.shape)
    logr_best_par = []
    for d in range(y_train.shape[1]):
        # custom scorer for logit-regression regressor:  flip (greater_is_better=False)
        # default mserror (metrics.mean_squared_error) computed on the prediction
        # probabilities (needs_proba=True) and not on the predictions of the model.
        custom_logr_mse_score = make_scorer(mean_squared_error, greater_is_better=False,
                                            needs_proba=True)
        logr_grid_d = GridSearchCV(logr_pipe, cv=cv_splits,
                                   param_grid=logr_param_grid_pipe,
                                   n_jobs=-1,
                                   scoring=custom_logr_mse_score
                                   # TODO: check
                                   )
        logr_grid_d.fit(x_train, y_train[:, d])
        logr_best_par.append(np.array([logr_grid_d.best_params_['LOGR__C']]))
        logr_y_pred[:, d] = logr_grid_d.predict(x_test[:, d])
    logr_best_par = np.median(logr_best_par, axis=1)  # C
    logr_gof = goodness_of_fit(y_test, logr_y_pred, verbose=0)
    return logr_y_pred, logr_gof, logr_best_par


def goodness_of_fit(Ytrue, Ypred, verbose=0):
    '''

    '''

    r2 = r2_score(Ytrue, Ypred,
                  multioutput="raw_values")  # multioutput alternatives: "uniform_variance" or "variance_weighted"
    r2 = np.mean(r2[r2 != 1])
    expl_var = explained_variance_score(Ytrue, Ypred, multioutput="uniform_average")
    # average magnitude of the error (same unit as y)
    mae = mean_absolute_error(Ytrue, Ypred, multioutput="uniform_average")
    # average squared magnitude of the error (same unit as y), penalizes larger errors
    rmse = np.mean(np.sqrt(mean_squared_error(Ytrue, Ypred, multioutput="raw_values")))
    with np.errstate(divide='ignore', invalid='ignore'):
        # nmae = mae normalized over range(Ytrue) (for each channel) and averaged over all the outputs.
        # Analogous to mae, normalized over the range of ytrue to compare it with the nmae of other datasets with possibly different ranges.
        # Can be seen as the "avg magnitude of the error is 13% of the amplitude of ytrue".
        nmae = np.mean(np.nan_to_num(
            np.divide(mean_absolute_error(Ytrue, Ypred, multioutput="raw_values"),
                      np.ptp(Ytrue, axis=0))))
        # nmrse = rmse normalized over var(Ytrue) (for each channel) and averaged over all the outputs.
        # Analogous to the rmse, normalized over the variance of ytrue to compare it with the nrmse of other datasets with possibly different variance.
        nrmse = np.mean(np.nan_to_num(
            np.divide(np.sqrt(mean_squared_error(Ytrue, Ypred, multioutput="raw_values")),
                      np.var(Ytrue, axis=0))))
    if verbose == 1:
        print("Coefficient of determination R2: " + str(r2))
        print("Explained Variance: " + str(expl_var))
        print("Mean Absolute Error: " + str(mae))
        print("Mean Squared Error: " + str(rmse))
        print("Range Normalized Mean Absolute Error: " + str(nmae))
        print("Variance Normalized Mean Squared Error: " + str(nrmse))
    return r2, expl_var, mae, rmse, nmae, nrmse


def idxmask2binmask(m, coeff=1):
    '''
    Converts an indices mask to a binary mask. The binary mask is == 1 in all
    the indices contained in the mask, and 0 otherwise.
    '''
    m = np.array(m)
    assert len(m.shape) == 1 or (
            len(m.shape) == 2 and (m.shape[0] == 1 or m.shape[1] == 1))
    if len(m.shape) == 2:
        m = m.ravel()
    assert m[0] >= 0 and all(np.diff(m) > 0)
    v = np.zeros(np.max(m) + 1)
    v[m] = 1 * coeff
    return v


# TODO: delete this
# def filter_data_columns(data):
#     '''
#     Filters out useless columns from the loaded database.
#     The fiels are described in cognolato20gaze, table 4.
#     Columns of interest and zero based original_idx:
#     * (re)grasp (7)
#     * (re)grasprepetition (8)
#     * (re)object (9)
#     * (re)position (12)
#     * (re)dynamic (13)
#     * emg (50-61)
#     * ts (103)
#     '''
#
#     data = data[:, (7, 8, 9, 12, 13, np.arange(50, 61), 103)]
#     return data


#############################################################
############################ MAIN ###########################
#############################################################
def main():
    ##### parse parameters #####
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datapath', required=True,
                        help='path to the database. Ex: path\\fname.mat. Path '
                             'must be relative to the .py script.')
    parser.add_argument('--trainreps', required=False, default='1 2 3',
                        help='training repetitions (from 1 to 4). Ex: 1 2 3')
    parser.add_argument('--testreps', required=False, default='4',
                        help='test repetitions (from 1 to 4). Ex: 4')
    parser.add_argument('--alltraintestsplits', required=False, default=False,
                        help='if true, overrides trainreps and testreps. Performance '
                             'will be computed over the average of all the possible '
                             'train-test splits 123-4,124-3,134-2,124-3. Default: False.')

    parser.add_argument('--hypopt', required=False, default=False,
                        help='hyperparameter optimization. Default: False.')

    parser.add_argument('--linearregressor', required=False, default="nonlinear",
                        help='whether to use RFF (nonlinear) or not (linear). Default: '
                             'nonlinear.')
    parser.add_argument('--numRFFseeds', required=False, default="1",
                        help='how many times to recompute RFFs. Default: 1.')

    parser.add_argument('--saveoutput', required=False,
                        help='if needed, path where to save the output file. Ex: '
                             'Results\\')
    args = parser.parse_args()

    fields = ['regrasp', 'regrasprepetition', 'reobject', 'reposition', 'redynamic',
              'emg', 'ts']
    data = loadmat(args.datapath, variable_names=fields)

    if not args.alltraintestsplits:
        train_reps = np.fromstring(args.trainreps, dtype=np.int, sep=' ')
        test_reps = np.fromstring(args.testreps, dtype=np.int, sep=' ')
    else:
        # combination of possible train-test reps
        reps = (((1, 2, 3), (4)), ((1, 2, 4), (3)), ((1, 3, 4), (2)), ((2, 3, 4), (1)))

    use_rff = args.linearregressor == 'nonlinear'
    n_rff = 300
    n_seeds_rff = 1  # the seeds will be range(num_seeds_rff)

    out_fname = ""
    if args.saveoutput:
        if use_rff == 1:
            rfftext = 'rff' + n_seeds_rff + '_'
        else:
            rfftext = ''
        out_fname = os.path.join(args.saveoutput, 'res_' +
                                 ntpath.basename(args.datapath).split('.')[0] + '_' +
                                 rfftext)

    hyp_opt = args.hypopt
    if hyp_opt == True:
        alphas_vec = np.logspace(-4, 4, 4)
        gammas_vec = np.logspace(-4, 4, 4)
        Cs_vec = np.logspace(-4, 4, 4)
        out_fname = out_fname + '_hypOpt'
    else:
        alphas_vec = [1.0]
        gammas_vec = [0.5]
        Cs_vec = [1.0]
        out_fname = out_fname + '_noHypOpt'

    # sys.stdout = open(out_fname + '.txt', 'w')  # redirect console to file

    ##### process database #####
    data['regrasp'] = convert_label(data['regrasp'])
    data = extract_features(data)

    # split database
    x_train, y_train, x_test, y_test, cv_splits = split_data(data, train_reps,
                                                             test_reps,
                                                             debug_plot = True,
                                                             subsample_train=True)

    ##### RFF-RR averaged over multiple seeds #####
    # rr_best_par_each_seed = []
    # rr_y_pred_each_seed = []
    # rr_gof_each_seed = []
    # for seed in range(n_seeds_rff):  # average results over multiple rff seeds
    #
    #     rr_pipe = Pipeline([
    #         ('RFF', RBFSampler(n_components=n_rff, random_state=seed)),
    #         ('RR', Ridge())
    #     ])
    #     rr_param_grid_pipe = [{'RFF__gamma': gammas_vec, 'RR__alpha': alphas_vec}, ]
    #     rr_grid = GridSearchCV(rr_pipe, cv=cv_splits,
    #                            param_grid=rr_param_grid_pipe,
    #                            scoring="r2", iid=False)
    #     rr_grid.fit(x_train, y_train)
    #     rr_best_par_each_seed.append(np.array([rr_grid.best_params_['RR__alpha'],
    #                                            rr_grid.best_params_['RFF__gamma']]))
    #     rr_y_pred_each_seed.append(rr_grid.predict(x_test))
    #     rr_gof_each_seed.append(goodness_of_fit(y_test, rr_y_pred_each_seed[-1],
    #                                             verbose=0))  # r2, expl_var, mae, rmse, nmae, nrmse
    #
    # # cumulative results rff for multiple seeds
    # rr_best_par = np.median(rr_best_par_each_seed, axis=0)  # alpha, gamma
    # rr_y_pred = np.mean(rr_y_pred_each_seed, axis=0)
    # rr_gof = np.mean(rr_gof_each_seed, axis=0)

    ##### logistic regression (one instance per dof) #####

    # custom scorer for logit-regression regressor:  flip (greater_is_better=False)
    # default mserror (metrics.mean_squared_error) computed on the prediction
    # probabilities (needs_proba=True) and not on the predictions of the model.
    custom_logr_mse_score = make_scorer(mean_squared_error, greater_is_better=False,
                                        needs_proba=True)
    logr_pipe = Pipeline([('dimred', None), ('LOGR', LogisticRegression(class_weight=None,
                                                                        random_state=1,
                                                                        n_jobs=None))])
    Cs_vec = [0.01,0.1,1,10] # TODO! REMOVE THIS, ADDED JUST FOR DEBUG
    logr_param_grid_pipe = {'LOGR__C': Cs_vec}
    logr_grid_d = []
    for d in range(y_train.shape[1]):  # for each dof
        logr_grid_d.append(GridSearchCV(logr_pipe, cv=cv_splits,
                                        param_grid=logr_param_grid_pipe,
                                        n_jobs=None, scoring=custom_logr_mse_score
                                        # TODO: check
                                        ))
        logr_grid_d[-1].fit(x_train, y_train[:, d])

    # determine best C overall

    # logr_y_pred, logr_gof, logr_best_par = multiple_log_reg(x_train, y_train, x_test,
    #                                                         y_test, cv_splits, logr_pipe,
    #                                                         logr_param_grid_pipe)

    print('The end')


if __name__ == '__main__':
    main()
