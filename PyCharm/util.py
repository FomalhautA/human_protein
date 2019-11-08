

import numpy as np
import pandas as pd

import os
import copy

import csv
import random
import tensorflow as tf
import matplotlib.pylab as plt
from PIL import Image

from pipline import LR_DECAY, ORIG_CHANNEL, NORM_AVG, NORM_STD


def mat_to_dict(ids, f_mat):
    """
    Convert id list and feature matrix into dictionary.
    """
    res = dict()
    for id_, f in zip(ids, f_mat):
        res[id_] = f

    return res


def get_name_set(filelist):
    """
    Fetch common name from file names in file name list.
    Common name does not contain color.

    filelist, list of file names.

    Return a set of common name.
    """
    color = ["green", "blue", "red", "yellow"]
    name_set = set()
    for filename in filelist:
        name = filename.split("_")[0]
        name_set.add(name)

    return name_set


# def get_image_tensor(idx, fname_lst):
#     """
#     Get image tensor with index "idx" from file name list "fname_lst".
#     idx, index of image
#     fname_lst, list of file names
#     """
#     fname = file_name_green[idx]
#     full_name = os.path.join(folder, fname)
#
#     image_data = tf.gfile.FastGFile(full_name, 'rb').read()
#     image_tensor = tf.image.decode_png(image_data, channels=3)
#
#     return image_tensor


# def get_image_tensors(fname_lst):
#     """
#     Get full dimension image tensor in given filename list.
#     fname_lst, list of file names.
#
#     Return image tensors.
#     """
#     image_tensors = tf.convert_to_tensor([get_image_tensor(idx)
#                                           for idx in range(len(fname_lst))], tf.float32)
#     print("image arr shape: ", image_tensors.get_shape())
#
#     return image_tensors


def normalizer(mat):
    for i in range(ORIG_CHANNEL):
        mat[:, i, :, :] = (mat[:, i, :, :] - NORM_AVG[i]) / NORM_STD[i]


def matrix_reshape(mat):
    temp = np.zeros((mat.shape[0], mat.shape[2], mat.shape[3], mat.shape[1]))
    for r in range(mat.shape[0]):
        for i in range(mat.shape[2]):
            for j in range(mat.shape[3]):
                for k in range(mat.shape[1]):
                    temp[r][i][j][k] = mat[r][k][i][j]

    return temp


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
            std_tmp = np.std(image_arr[j])
            std_lst[j] = inc_std(std_lst[j], i+1, avg_lst[j], std_tmp)
    print([round(item, 2) for item in std_lst])

    return avg_lst, std_lst


def inc_avg(avg, N, x):

    if N >= 1:
        return avg*(N-1)/N + x/N
    else:
        raise Exception('N must be zero or positive integer.')


def inc_std(std, N, avg, x):

    if N >= 1:
        return (std**2*(N-1)/N + (x-avg)**2/N)**0.5
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
        temp = []
        for channel in ['red', 'green', 'blue', 'yellow']:
            im = Image.open(os.path.join(folder, fname+'_'+channel+'.png'), 'r')
            temp.append(np.array(im.convert('L')))

        img_arr.append(np.array(temp))

    return np.array(img_arr)


def rand_sample(f_dict, scale):
    """
    Randomly select samples from dict, filename as key, image data as values.
    Scale is sample numbers you want.
    """
    subset = dict()
    keys = random.sample(f_dict.keys(), scale)
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


def fetch_batch_X(f_dict, scale, idx, folder):
    pick_list = list(f_dict.keys())

    fname_batch_ = pick_list[idx * scale: (idx + 1) * scale]
    # fname_green_batch_ = [item + "_green.png" for item in fname_batch_]

    image_arr = get_image_arr(fname_batch_, folder)

    return image_arr


def fetch_batch_Y(f_dict, scale, idx=0):
    pick_list = list(f_dict.keys())

    fname_batch_ = pick_list[idx * scale: (idx + 1) * scale]
    # fname_green_batch_ = [item + "_green.png" for item in fname_batch_]

    labels_batch_ = pick_labels(f_dict, fname_batch_)
    labels = labels_batch_
    labels = labels.reshape((-1, 1, 1, 28))

    return labels


def fetch_batch_data(f_dict, scale, idx, folder):
    """
    f_dict, filename dictionary
    scale, batch size
    idx, batch number

    """
    X = fetch_batch_X(f_dict, scale, idx, folder)
    Y = fetch_batch_Y(f_dict, scale, idx)

    return X, Y


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

    W_p = scale / N_p

    W_n = scale / N_n

    # W_p = np.array([item if item < threshold else threshold for item in W_p])
    #
    # W_n = np.array([item if item < threshold else threshold for item in W_n])
    #
    return W_p.reshape(1, -1), W_n.reshape(1, -1), N_p, N_n


def showWeights(W_p, W_n, N_p, N_n):
    """
    Show weighted weights W_p, W_n and Count N_p, N_n.
    """
    print([round(item, 6) for item in W_p[0]])
    print([round(item, 6) for item in W_n[0]])
    print([int(item) for item in N_p])
    print([int(item) for item in N_n])


def plot_performance(epochs, cost=None, acc=None, rr=None, pr=None, f1=None,
                     step=1, batch_cnt=10, save_name="result.png"):
    """
    Visulizing performance index evolving during trainnig.
    epochs, trainning epochs
    cost, 1-d array for loss
    acc, 1-d array for accuracy
    rr, 1-d array for recall rate
    pr, 1-d array for precise rate
    f1, 1-d array for f1 score
    """
    count = epochs // step * batch_cnt
    x = np.linspace(1, epochs, count)

    fig = plt.figure(figsize=(15, 8))
    if cost:
        plt.subplot(2, 3, 1)
        plt.plot(x, cost)
        plt.title("Total Weighted Loss")

    if acc:
        plt.subplot(2, 3, 2)
        plt.plot(x, acc)
        plt.ylim([0, 1])
        plt.title("Averaged Accuracy")

    if rr:
        plt.subplot(2, 3, 3)
        plt.plot(x, rr)
        plt.ylim([0, 1])
        plt.title('Averaged Recall Rate')

    if pr:
        plt.subplot(2, 3, 4)
        plt.plot(x, pr)
        plt.ylim([0, 1])
        plt.title('Averaged Precision Rate')

    if f1:
        plt.subplot(2, 3, 5)
        plt.plot(x, f1)
        plt.ylim([0, 1])
        plt.title('Averaged F1-Score')

    fig.savefig(save_name)

    # plt.show()


def my_decay(a, b):
    """
    Exponential decay function with decay_rate, decay_steps.
    """
    return tf.train.exponential_decay(a, b, decay_steps=2, decay_rate=LR_DECAY, staircase=True)


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
        return ""
    res = ""
    for item in hotcode:
        res += str(item)
        res += " "

    return res[:-1]


def dict_substract(dict1, dict2):
    for key in dict2.keys():
        del dict1[key]

    return dict1


def load_data(tv_ratio=0.1, ratio=0.002):
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
    f_dict_train = dict_substract(f_dict, f_dict_val)

    return f_dict_train, f_dict_val, f_dict_full


def batch_data_generator(f_dict, batch_size):
    """

    :param f_dict:
    :param batch_size:
    :return:
    """
    scale = len(f_dict.keys()) if len(f_dict.keys()) < batch_size else batch_size

    batch_dict = rand_sample(f_dict, scale)
    f_dict = dict_substract(f_dict, batch_dict)

    X_train, Y_train = fetch_batch_data(batch_dict, scale, 0, "../Data/train_s")

    normalizer(X_train)
    X_train = matrix_reshape(X_train)

    return X_train, Y_train


# def predict(sess, logits, X, test_flst_n):
#     resfile = open("./Data/result.csv", "w", newline="")
#     res_writer = csv.writer(resfile)
#     res_writer.writerow(["Id", "Predicted"])
#     for fname in test_flst_n:
#         image_arr = get_image_arr([fname+"_green.png"], "./Data/test_s")
#
#         y_test_ = sess.run(logits, feed_dict={X: image_arr})
#         y_test_ = list(np.squeeze(y_test_))
#
#         temp = []
#         for i, item in enumerate(y_test_):
#             if item > 0.5:
#                 temp.append(i)
#
#         predict = convert_res_str(temp)
#
#         res_writer.writerow([fname, predict])
#
#     resfile.close()


def model_calc(sess, logits, X, f_dict):
    calc_batch = 64
    inner_loop = len(f_dict.keys()) // calc_batch

    y_p = []
    labels = []
    for i in range(inner_loop):
        X_data, Y_data = batch_data_generator(f_dict, calc_batch)
        y_ = sess.run(logits, feed_dict={X: X_data})
        y_ = np.squeeze(y_)
        y_p.append(np.array(y_ > 0.5, dtype='int'))
        labels.append(Y_data)

    if inner_loop * calc_batch < len(f_dict.keys()):
        X_data, Y_data = batch_data_generator(f_dict, calc_batch)
        y_ = sess.run(logits, feed_dict={X: X_data})
        y_ = np.squeeze(y_)
        y_p.append(np.array(y_ > 0.5, dtype='int'))
        labels.append(Y_data)

    Y_ = y_p[0]
    for i in range(1, len(y_p)):
        Y_ = np.vstack([Y_, y_p[i]])

    Y = labels[0]
    for i in range(1, len(labels)):
        Y = np.vstack([Y, labels[i]])

    return Y_, Y
