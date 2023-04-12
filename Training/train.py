import numpy as np
import pandas as pd
import tensorflow as tf
import math
import random
import os, time, argparse
from optparse import OptionParser

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler,CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from model import model_multiview_aafeature_bert


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_accuracy = accuracy_score(val_targ, val_predict)

        logs['val_accuracy'] = _val_accuracy
        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision

        print(" — val_f1: %f — val_precision: %f — val_recall: %f — val_accuracy: %f" % (
        _val_f1, _val_precision, _val_recall, _val_accuracy))
        return

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 50.0
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))
    return lrate


def seed_tensorflow(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_model_multiview_aafeature_bert(path, train_bert, train_aa_fea, y_train, test_bert, test_aa_fea, y_test):
    parameters_string = "DeepAFP"
    train_path = os.path.join(path, parameters_string)
    os.makedirs(train_path)
    param_distribution = {
        "filter_num": [32, 64],
        "kernel_size": [3, 4, 5, 6, 7, 8],
        "lstm_num": [20, 40, 60]
    }
    kf = KFold(n_splits=5, shuffle=True)

    with open(os.path.join(train_path, "cross_validation.txt"), "w") as fww:
        fww.write("Parameter\tFold\tAccuracy\n")
        with open(os.path.join(train_path, "search_process.txt"), "w") as fw:
            best_score, best_filter_num, best_kernel_size, best_lstm_num = 0, 0, 0, 0
            for filter_num in param_distribution["filter_num"]:
                for kernel_size in param_distribution["kernel_size"]:
                    for lstm_num in param_distribution["lstm_num"]:
                        all_acc_score = []
                        for i, (train_index, val_index) in enumerate(kf.split(train_aa_fea)):
                            X_train_bert, X_val_bert = train_bert[train_index], train_bert[val_index]
                            X_train_aafea, X_val_aafea = train_aa_fea[train_index], train_aa_fea[val_index]
                            Y_train, Y_val = y_train[train_index], y_train[val_index]
                            adam = Adam(learning_rate=1e-3)
                            network = model_multiview_aafeature_bert(filter_num=filter_num, kernel_size=kernel_size,
                                                                     lstm_num=lstm_num)
                            lrate = LearningRateScheduler(step_decay)
                            Early = EarlyStopping(monitor="val_accuracy", mode='max', patience=20, verbose=2,
                                                  restore_best_weights=True)
                            network.fit(x=[X_train_bert, X_train_aafea], y=Y_train, batch_size=128, epochs=300,
                                        callbacks=[Metrics(valid_data=([X_val_bert, X_val_aafea], Y_val)), lrate,
                                                   Early])
                            y_pred = np.argmax(network.predict([X_val_bert, X_val_aafea]), -1)
                            acc = accuracy_score(np.argmax(Y_val, -1), y_pred)
                            all_acc_score.append(acc)
                            fww.write(
                                "filter_num={},kernel_size={},lstm_num={}\t{}\t{}\n".format(filter_num, kernel_size,
                                                                                            lstm_num, str(i), str(acc)))

                        score = np.mean(all_acc_score)
                        if score > best_score:
                            best_score, best_filter_num, best_kernel_size, best_lstm_num = score, filter_num, kernel_size, lstm_num
                        fw.write(
                            "filter_num={}\tkernel_size={}\tlstm_num={}\tAccuracy={}\n".format(filter_num, kernel_size,
                                                                                               lstm_num, score))

            Early = EarlyStopping(monitor="val_accuracy", mode='max', patience=20, verbose=2, restore_best_weights=True)
            network = model_multiview_aafeature_bert(filter_num=best_filter_num, kernel_size=best_kernel_size,
                                                     lstm_num=best_lstm_num)
            lrate = LearningRateScheduler(step_decay)
            csv_logger = CSVLogger(os.path.join(train_path, "model_training.csv"))
            network.fit(x=[train_bert, train_aa_fea], y=y_train, batch_size=128, epochs=300,
                        callbacks=[Metrics(valid_data=([test_bert, test_aa_fea], y_test)), lrate, csv_logger, Early])
            network.save(os.path.join(train_path, "best_model.h5"))
            y_pred = np.argmax(network.predict([test_bert, test_aa_fea]), -1)
            acc = accuracy_score(np.argmax(y_test, -1), y_pred)
            recall = recall_score(np.argmax(y_test, -1), y_pred)
            prec = precision_score(np.argmax(y_test, -1), y_pred)
            f1 = f1_score(np.argmax(y_test, -1), y_pred)
            auc = roc_auc_score(np.argmax(y_test, -1), y_pred)
            with open(os.path.join(train_path, "best_parameter.txt"), "w") as fw1:
                fw1.write(
                    "best_filter_num={}\tbest_kernel_size={}\tbest_lstm_num={}\tacc={}\trecall={}\tprec={}\tf1={}\tauc={}\n".format(
                        best_filter_num, best_kernel_size, best_lstm_num, acc, recall, prec, f1, auc))

if __name__ == "__main__":
    hstr = "Train AFP Prediction"
    parser = OptionParser(hstr, description='Train AFP Prediction')

    parser.add_option('--seed', action='store', type=int, default=810, dest='seed',
                      help='random seed severed for training phase')

    (options, args) = parser.parse_args()
    RANDOM_SEED = options.seed
    seed_tensorflow(seed=RANDOM_SEED)

    # Firstly, import features of peptides on train set, including bert, binary profile, blosum62 and z-scale
    # ==================================================================================
    train_bert = np.load('/share/home/grp-lizy/pangyx/Experiments/AFP/data/feature/dataset1-train.npy',
                         allow_pickle=True)
    train_bin = np.load('/share/home/grp-lizy/pangyx/Experiments/AFP/data/feature/dataset1-train_bin.npy',
                        allow_pickle=True)
    train_blo = np.load('/share/home/grp-lizy/pangyx/Experiments/AFP/data/feature/dataset1-train_blo.npy',
                        allow_pickle=True)
    train_zsl = np.load('/share/home/grp-lizy/pangyx/Experiments/AFP/data/feature/dataset1-train_zsl.npy',
                        allow_pickle=True)
    # ==================================================================================
    train_aa_fea = np.concatenate((train_bin, train_blo, train_zsl), axis=2)
    train_bin_blo = np.concatenate((train_bin, train_blo), axis=2)
    train_bin_zsl = np.concatenate((train_bin, train_zsl), axis=2)
    train_blo_zsl = np.concatenate((train_blo, train_zsl), axis=2)

    # Secondly, import features of peptides on test set, including bert, binary profile, blosum62 and z-scale
    # ==================================================================================
    test_bert = np.load('/share/home/grp-lizy/pangyx/Experiments/AFP/data/feature/dataset1-test.npy', allow_pickle=True)
    test_bin = np.load('/share/home/grp-lizy/pangyx/Experiments/AFP/data/feature/dataset1-test_bin.npy',
                       allow_pickle=True)
    test_blo = np.load('/share/home/grp-lizy/pangyx/Experiments/AFP/data/feature/dataset1-test_blo.npy',
                       allow_pickle=True)
    test_zsl = np.load('/share/home/grp-lizy/pangyx/Experiments/AFP/data/feature/dataset1-test_zsl.npy',
                       allow_pickle=True)
    # ==================================================================================
    test_aa_fea = np.concatenate((test_bin, test_blo, test_zsl), axis=2)
    test_bin_blo = np.concatenate((test_bin, test_blo), axis=2)
    test_bin_zsl = np.concatenate((test_bin, test_zsl), axis=2)
    test_blo_zsl = np.concatenate((test_blo, test_zsl), axis=2)

    # Thirdly, import labels on train and test sets
    df_train = pd.read_csv("/share/home/grp-lizy/pangyx/Experiments/AFP/data/dataset1-train.csv")
    df_test = pd.read_csv("/share/home/grp-lizy/pangyx/Experiments/AFP/data/dataset1-test.csv")
    # ==================================================================================

    value = df_train["label"].tolist()
    value_test = df_test["label"].tolist()
    value = to_categorical(value, num_classes=2)
    value = np.array(value)
    y_test = np.array(to_categorical(value_test, num_classes=2))
    y_train = value

    time_now = int(round(time.time() * 1000))
    time_now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time_now / 1000))

    task_dir = "DeepAFP_main_{}".format(time_now)
    os.makedirs(task_dir)
    print("=============Loading Data Successfully!===========", flush=True)

    train_model_multiview_aafeature_bert(task_dir, train_bert, train_aa_fea, y_train, test_bert, test_aa_fea, y_test)




