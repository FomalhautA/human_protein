import numpy as np
import pandas as pd

import csv
import os
import copy

import random
from PIL import Image

from pipline import ORIG_CHANNEL

NORM_AVG_GL = [20.58, 13.46, 14., 21.12]
NORM_STD_GL = [35.06, 25.9, 39.23, 35.32]
NORM_AVG_TEST = [15.12, 11.6, 10.4, 15.13]
NORM_STD_TEST = [30.07, 24.21, 33.08, 29.5]


def mat_to_dict(ids, f_mat):
    """
    Convert id list and feature matrix into dictionary.
    """
    res = dict()
    for id_, f in zip(ids, f_mat):
        res[id_] = f

    return res


def normalizer(mat, norm_avg, norm_std):
    mat = mat.astype(float)
    for i in range(ORIG_CHANNEL):
        mat[i] = (mat[i] - norm_avg[i]) / norm_std[i]

    return mat


def channel_norm_params(namelst, folder):

    avg_lst = [0, 0, 0, 0]
    for i, name in enumerate(namelst):
        image_arr = get_image_arr([name], folder)[0]
        for j in range(ORIG_CHANNEL):
            avg_tmp = np.mean(image_arr[j])

            avg_lst[j] = inc_avg(avg_lst[j], i+1, avg_tmp)
    print([round(item, 2) for item in avg_lst])

    std_lst = [0, 0, 0, 0]
    for i, name in enumerate(namelst):
        image_arr = get_image_arr([name], folder)[0]
        for j in range(ORIG_CHANNEL):
            std_tmp = np.sqrt(np.mean((image_arr[j]-avg_lst[j])**2))
            std_lst[j] = inc_std(std_lst[j], i+1, std_tmp)
    print([round(item, 2) for item in std_lst])

    return avg_lst, std_lst


def inc_avg(avg, N, x):

    if N >= 1:
        return avg*(N-1)/N + x/N
    else:
        raise Exception('N must be zero or positive integer.')


def inc_std(std, N, x):

    if N >= 1:
        return (std**2*(N-1)/N + x**2/N)**0.5
    else:
        raise Exception('N must be zero or positive integer.')


def get_image_arr(fname_lst, folder):
    """
    Get full dimension image array in given filename list.
    fname_lst, list of file names.

    Return image array.
    """
    img_arr = []
    for fname in fname_lst:
        img_arr.append(get_image(fname, folder))

    return np.array(img_arr)


def get_image(fname, folder):
    temp = []
    for channel in ['red', 'green', 'blue', 'yellow']:
        im = Image.open(os.path.join(folder, fname + '_' + channel + '.png'), 'r')
        temp.append(np.array(im.convert('L')))

    return np.array(temp)


def rand_sample(f_dict, scale):
    """
    Randomly select samples from dict, filename as key, image data as values.
    Scale is sample numbers you want.
    """
    subset = dict()

    keys = random.sample(list(f_dict.keys()), scale)
    for item in keys:
        subset[item] = f_dict[item]

    return subset


def pick_labels(f_dict, fname_lst):
    """
    Pick labels from f_dict.
    The picked feature matrix corresponding to file in fname_lst.

    f_dict, diction with fname as key and label vector as value.
    fname_lst, list of file names.

    Return label vectors.
    """
    res = []
    for fname in fname_lst:
        res.append(f_dict[fname])

    #     print("{} feature vector picked: ".format(len(res)))

    return np.array(res)


def fetch_data_x(fname, folder, norm_avg, norm_std):
    return np.transpose(normalizer(get_image(fname, folder), norm_avg=norm_avg, norm_std=norm_std), (1, 2, 0))


def fetch_data_y(dataframe, fname):
    return np.array(dataframe[fname]).reshape((1, 1, 28))


def get_weights(labels, scale=10, delta=1, threshold=0.004):
    """
    Calculate weights for positive samples and negative samples.
    labels, label vectors
    delta, small number to avoid divided by zero
    threshold, maximum weight threshold

    Return weight for positive, negative; count for positive, negative.
    """
    samples = np.squeeze(labels)

    (N, N_c) = samples.shape  # sample scale, classes

    N_p = np.sum(samples, axis=0)

    N_n = N - N_p

    W_p = N_n / N

    # W_p = np.array([item if item < threshold else threshold for item in W_p])
    #
    # W_n = np.array([item if item < threshold else threshold for item in W_n])
    #
    return W_p, 1 - W_p, N_p, N_n


def showWeights(W_p, W_n, N_p, N_n):
    """
    Show weighted weights W_p, W_n and Count N_p, N_n.
    """
    print([round(item, 6) for item in W_p[0]])
    print([round(item, 6) for item in W_n[0]])
    print([int(item) for item in N_p])
    print([int(item) for item in N_n])


def label_encode(target, cnumber=28):
    """
    target, 2-d list of string
    """
    r = len(target)
    if r == 0:
        return []

    res = np.zeros([r, cnumber])
    for i, str_lst in enumerate(target):
        for item in str_lst:
            col = int(item)
            res[i][col] = 1

    return res.astype("int")


def convert_res_str(hotcode):
    if len(hotcode) == 0:
        return "0"
    res = ""
    for item in hotcode:
        res += str(item)
        res += " "

    return res[:-1]


def dict_substract(dict1, dict2):
    for key in dict2.keys():
        del dict1[key]


def load_test_fname(folder):
    test_f_lst = os.listdir(folder)
    test_f_set = set([item.split("_")[0] for item in test_f_lst])
    test_flst = list(test_f_set)

    return test_flst


def data_partition(tv_ratio=0.1, ratio=0.002):
    """

    :param tv_ratio: train validation ratio
    :param ratio: ratio of picked data
    :return: dict of train set and dict of validation set
    """

    df = pd.read_csv("../Data/train.csv")
    df.head()

    ids = df.Id
    target = [item.split(' ') for item in df.Target]

    label_mat = label_encode(target)

    f_dict = mat_to_dict(ids, label_mat)
    total = len(ids)
    picked = int(total * ratio)

    f_dict = rand_sample(f_dict, picked)
    f_dict_full = copy.deepcopy(f_dict)
    f_dict_val = rand_sample(f_dict, int(picked*tv_ratio))
    dict_substract(f_dict, f_dict_val)
    f_dict_train = f_dict

    df_val = pd.DataFrame()
    for key in f_dict_val.keys():
        df_val[key] = f_dict_val[key]

    df_val.to_csv('../Data/val.csv', index=False)

    df_train = pd.DataFrame()
    for key in f_dict_train.keys():
        df_train[key] = f_dict_train[key]

    df_train.to_csv('../Data/tra.csv', index=False)

    df_full = pd.DataFrame()
    for key in f_dict_full.keys():
        df_full[key] = f_dict_full[key]

    df_full.to_csv('../Data/full.csv', index=False)

    # return len(f_dict_train.keys()), len(f_dict_val.keys()), len(f_dict_full.keys())
    return f_dict_train, f_dict_val, f_dict_full


def load_data():
    f_dict_train = pd.read_csv('../Data/tra.csv')
    f_dict_val = pd.read_csv('../Data/val.csv')
    f_dict_full = pd.read_csv('../Data/full.csv')

    return f_dict_train, f_dict_val, f_dict_full


def train_data_generator(f_dict, folder):
    while True:
        f_dict = rand_sample(f_dict, len(f_dict.keys()))
        for key in f_dict.keys():
                yield fetch_data_x(key, folder, NORM_AVG_GL, NORM_STD_GL), fetch_data_y(f_dict, key)


def eval_data_generator(f_dict, folder):
    while True:
        for key in f_dict.keys():
            yield fetch_data_x(key, folder, NORM_AVG_GL, NORM_STD_GL), fetch_data_y(f_dict, key)


def pred_data_generator(test_flst, folder):
    for key in test_flst:
        yield fetch_data_x(key, folder, NORM_AVG_TEST, NORM_STD_TEST)


def convert_y(y_):
    temp = []
    for i, item in enumerate(y_):
        if item > 0.5:
            temp.append(i)

    predict = convert_res_str(temp)

    return predict


def save_to_file(fnames, predictions):
    resfile = open("./performance/result.csv", "w", newline="")
    res_writer = csv.writer(resfile)
    res_writer.writerow(["Id", "Predicted"])
    ans = dict()
    for fname, pred in zip(fnames, predictions):
        ans[fname] = convert_y(pred)

    for key in sorted(ans.keys()):
        res_writer.writerow([key, ans[key]])

    resfile.close()


def f1_score(rr, pr):
    return 2*rr*pr/(rr+pr)