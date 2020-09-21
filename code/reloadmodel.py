"""
Example call: python reloadmodel.py --datapath ..\data\S010_ex1.mat --modelpath ..\results\test_2_subsampled\cnn_5_1600595245 --rebalanceclasses yes --window 400 --stride 200
"""

import argparse
import os
import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import deep_grasp_regression as dgr
import myutils_tensorflow as tf_utils


def main():
    # region parse parameters

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # dataset args
    parser.add_argument("--datapath", required=True,
                        help="path to the database. Ex: path\\fname.mat.")
    parser.add_argument("--modelpath", required=True,
                        help="path to the directory containig the tensorflo model."
                             "Ex: path_to_model_dir\\")
    parser.add_argument("--rebalanceclasses", required=False, default="no",
                        choices=["yes", "no"],
                        help="whether to rebalance training classes or not (yes/no). "
                             "Default: yes")
    parser.add_argument("--trainreps", required=False, default="1 2 3",
                        help="training repetitions (from 1 to 4). Ex: 1 2 3")
    parser.add_argument("--testreps", required=False, default="4",
                        help="test repetitions (from 1 to 4). Ex: 4")
    parser.add_argument("--subsamplerate", required=False, default=1, type=int,
                        help="how much to subsample the training set. A value v corresponds"
                             "to a subsampling of 1:v. Default: 1.")
    parser.add_argument("--shuffle", required=False, default=False, type=bool,
                        help="whether to shuffle the data or not. Default: False.")
    # transform x args
    parser.add_argument("--window", required=False, default=1, type=int,
                        help="dimension of sliding window. Default: 1.")
    parser.add_argument("--stride", required=False, default=1, type=int,
                        help="stride of the sliding window. Default: 1.")
    # store args
    parser.add_argument("--saveoutput", required=False, default=None,
                        help="if needed, path where to save the output file. Ex: "
                             ".\\results\\")
    args = parser.parse_args()

    # parse args
    train_reps = np.fromstring(args.trainreps, dtype=np.int, sep=" ")
    test_reps = np.fromstring(args.testreps, dtype=np.int, sep=" ")
    feature_set = ["raw"]
    w_len = args.window
    w_stride = args.stride

    # endregion

    # region load and process data

    # load megane pro data
    fields = ["regrasp", "regrasprepetition", "reobject", "reposition", "redynamic",
              "emg", "ts"]
    x_train, y_train, x_test, y_test, cv_splits = \
        dgr.load_megane_database(args.datapath, fields, train_reps, test_reps,
                                 args.subsamplerate, feature_set,
                                 rebalance_classes=(args.rebalanceclasses == "yes"))

    x_val = x_train[cv_splits[0][1]]
    y_val = y_train[cv_splits[0][1]]
    x_train = x_train[cv_splits[0][0]]
    y_train = y_train[cv_splits[0][0]]

    # convert to tensorflow database
    n_features = x_train.shape[1]
    n_targets = y_train.shape[1]
    batch_size = 128
    shuffle_data = (args.shuffle == "yes")
    shuffle_buffer_size = 1000

    args_window_fn = [x_train, y_train, w_len, w_stride]  # arguments of the generator fn
    db_train = tf.data.Dataset.from_generator(
        generator=tf_utils.create_sliding_window_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([w_len, n_features]), n_targets),
        args=args_window_fn)
    # quick_visualize_img_tf_db_unbatched(db_train, num_images=10)  # for debug
    db_train = db_train.map(lambda x, y: (tf.expand_dims(x, axis=-1), y))
    if shuffle_data:
        db_train = db_train.shuffle(buffer_size=shuffle_buffer_size, seed=1)
    # num_complete_batches_train = tf_utils.get_db_shape(db_train, batched=False)[0] // batch_size
    db_train = db_train.batch(batch_size)

    args_window_fn = [x_val, y_val, w_len, w_stride]
    db_val = tf.data.Dataset.from_generator(
        generator=tf_utils.create_sliding_window_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([w_len, n_features]), n_targets),
        args=args_window_fn)
    # quick_visualize_img_tf_db_unbatched(db_val, num_images=10)  # for debug
    db_val = db_val.map(lambda x, y: (tf.expand_dims(x, axis=-1), y))
    if shuffle_data:
        db_val = db_val.shuffle(buffer_size=shuffle_buffer_size, seed=1)
    # num_complete_batches_val = tf_utils.get_db_shape(db_val, batched=False)[0] // batch_size
    db_val = db_val.batch(batch_size)

    args_window_fn = [x_test, y_test, w_len, w_stride]
    db_test = tf.data.Dataset.from_generator(
        generator=tf_utils.create_sliding_window_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([w_len, n_features]), n_targets),
        args=args_window_fn)
    # quick_visualize_img_tf_db_unbatched(db_test, num_images=10)  # for debug
    db_test = db_test.map(lambda x, y: (tf.expand_dims(x, axis=-1), y))
    # if shuffle_data:
    #     db_test = db_test.shuffle(buffer_size=shuffle_buffer_size, seed=1)
    db_test = db_test.batch(batch_size)

    # endregion

    # region compute baseline

    # baseline = predicting mean value for each dof of the training data
    m = np.mean(np.vstack([y for _, y in db_train]), axis=0)[None, :]
    pred_shape = tf_utils.get_db_shape(db_test, batched=True)[1]
    bsl_pred = np.multiply(np.ones(pred_shape), m)
    _, y_test = tf_utils.get_db_elems_labels(db_test)
    bsl_mae = np.mean(np.abs(y_test - bsl_pred), axis=0)

    # endregion


    # region reload model

    model = tf.keras.models.load_model(args.modelpath)

    # endregion

    # region perform predictions

    y_pred, mae = dgr.predict_evaluate_model(model, db_test)
    fig = dgr.quick_visualize_vec(y_test, y_pred)
    dgr.quick_visualize_vec(bsl_pred, continue_on_fig=fig,
                            title=f"y_test vs y_pred, {model.name},\n"
                                  f"mae={mae.ravel()}\n"
                                  f"bsl_mae={bsl_mae.ravel()}\n")
    if args.saveoutput is not None:
        fig.savefig(os.path.join(args.saveoutput, f"pred_{model.name}.png"), format="png")
    plt.show()

    # endregion


if __name__ == "__main__":
    main()
