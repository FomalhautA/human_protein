import numpy as np
import pandas as pd

import csv
import os
import copy

import random
from PIL import Image

from pipline import ORIG_CHANNEL
from threadsafe_iter import threadsafe_generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataUtils:
    def __init__(self):
        # NORM_AVG_GL = [20.58, 13.46, 14., 21.12]
        # NORM_STD_GL = [35.06, 25.9, 39.23, 35.32]

        self.NORM_AVG_GL = [21.35, 12.74, 14.38, 21.65]
        self.NORM_STD_GL = [35.5, 24.61, 39.67, 35.76]
        self.NORM_AVG_TEST = [15.12, 11.6, 10.4, 15.13]
        self.NORM_STD_TEST = [30.07, 24.21, 33.08, 29.5]

        self.COLORS = ['red', 'green', 'blue', 'yellow']

    @staticmethod
    def mat_to_dict(ids, f_mat):
        """
        Convert id list and feature matrix into dictionary.
        """
        res = dict()
        for id_, f in zip(ids, f_mat):
            res[id_] = f

        return res

    @staticmethod
    def normalizer(mat, norm_avg, norm_std):
        mat = mat.astype(float)
        for i in range(ORIG_CHANNEL):
            mat[i] = (mat[i] - norm_avg[i]) / norm_std[i]

        return mat

    def channel_norm_params(self, namelst, folder):

        avg_lst = [0, 0, 0, 0]
        for i, name in enumerate(namelst):
            image_arr = self.get_image_arr([name], folder)[0]
            for j in range(ORIG_CHANNEL):
                avg_tmp = np.mean(image_arr[j])

                avg_lst[j] = self.inc_avg(avg_lst[j], i+1, avg_tmp)
        print([round(item, 2) for item in avg_lst])

        std_lst = [0, 0, 0, 0]
        for i, name in enumerate(namelst):
            image_arr = self.get_image_arr([name], folder)[0]
            for j in range(ORIG_CHANNEL):
                std_tmp = np.sqrt(np.mean((image_arr[j]-avg_lst[j])**2))
                std_lst[j] = self.inc_std(std_lst[j], i+1, std_tmp)
        print([round(item, 2) for item in std_lst])

        return avg_lst, std_lst

    @staticmethod
    def inc_avg(avg, N, x):

        if N >= 1:
            return avg*(N-1)/N + x/N
        else:
            raise Exception('N must be zero or positive integer.')

    @staticmethod
    def inc_std(std, N, x):

        if N >= 1:
            return (std**2*(N-1)/N + x**2/N)**0.5
        else:
            raise Exception('N must be zero or positive integer.')

    def get_image_arr(self, fname_lst, folder):
        """
        Get full dimension image array in given filename list.
        fname_lst, list of file names.

        Return image array.
        """
        img_arr = []
        for fname in fname_lst:
            img_arr.append(self.get_image(fname, folder))

        return np.array(img_arr)

    def get_image(self, fname, folder, aug=True):

        temp_name = fname
        if not aug:
            temp_name = fname.split('-aug')[0] if fname.__contains__('-aug') else fname

        temp = []
        for channel in self.COLORS:
            im = Image.open(os.path.join(folder, temp_name + '_' + channel + '.png'), 'r')
            # temp.extend(np.transpose(np.array(im.convert('RGB')), axes=(2, 0, 1)))
            temp.append(np.asarray(im))

        return np.array(temp)

    @staticmethod
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

    @staticmethod
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

    def fetch_data_x(self, fname, folder, norm_avg, norm_std):
        return np.transpose(self.get_image(fname, folder)/255., (1, 2, 0))

    @staticmethod
    def fetch_data_y(dataframe, fname):
        return np.array(dataframe[fname])

    @staticmethod
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

    @staticmethod
    def showWeights(W_p, W_n, N_p, N_n):
        """
        Show weighted weights W_p, W_n and Count N_p, N_n.
        """
        print([round(item, 4) for item in W_p])
        print([round(item, 4) for item in W_n])
        print([int(item) for item in N_p])
        print([int(item) for item in N_n])

    @staticmethod
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

    @staticmethod
    def convert_res_str(hotcode):
        if len(hotcode) == 0:
            return "0"
        res = ""
        for item in hotcode:
            res += str(item)
            res += " "

        return res[:-1]

    @staticmethod
    def dict_substract(dict1, dict2):
        for key in dict2.keys():
            del dict1[key]

    @staticmethod
    def load_test_fname(folder):
        test_f_lst = os.listdir(folder)
        test_f_set = set([item.split("_")[0] for item in test_f_lst])
        test_flst = list(test_f_set)

        return test_flst

    def data_partition(self, tv_ratio=0.1, ratio=0.002):
        """

        :param tv_ratio: train validation ratio
        :param ratio: ratio of picked data
        :return: dict of train set and dict of validation set
        """

        df = pd.read_csv("../Data/train.csv")
        df.head()

        ids = df.Id
        target = [item.split(' ') for item in df.Target]

        label_mat = self.label_encode(target)

        f_dict = self.mat_to_dict(ids, label_mat)
        total = len(ids)
        picked = int(total * ratio)

        f_dict_p = self.rand_sample(f_dict, picked)
        f_dict_full = copy.deepcopy(f_dict_p)
        f_dict_val = self.rand_sample(f_dict_p, int(picked*tv_ratio))
        self.dict_substract(f_dict_p, f_dict_val)
        f_dict_train = f_dict_p

        df_val = pd.DataFrame()
        df_val = df_val.from_dict(f_dict_val)
        df_val.to_csv('../Data/val.csv', index=False)

        df_train = pd.DataFrame()
        df_train = df_train.from_dict(f_dict_train)
        df_train.to_csv('../Data/tra.csv', index=False)

        df_full = pd.DataFrame()
        df_full = df_full.from_dict(f_dict_full)
        df_full.to_csv('../Data/full.csv', index=False)

        return f_dict_train, f_dict_val, f_dict_full

    @staticmethod
    def load_data(train='../Data/tra.csv', val='../Data/val.csv', full='../Data/full.csv'):
        f_dict_train = pd.read_csv(train)
        f_dict_val = pd.read_csv(val)
        f_dict_full = pd.read_csv(full)

        return f_dict_train, f_dict_val, f_dict_full

    # @threadsafe_generator
    def train_data_generator(self, f_dict, folder, batch_size):
        while True:
            idx = 0
            total = len(f_dict.keys())
            f_dict = self.rand_sample(f_dict, total)
            while idx + batch_size <= total:
                X, Y = [], []
                for key in list(f_dict.keys())[idx:idx+batch_size]:
                    X.append(self.fetch_data_x(key, folder, self.NORM_AVG_GL, self.NORM_STD_GL))
                    Y.append(self.fetch_data_y(f_dict, key))

                idx += batch_size
                yield np.array(X), np.array(Y)

    # @threadsafe_generator
    def eval_data_generator(self, f_dict, folder):
        while True:
            for key in f_dict.keys():
                # yield fetch_data_x(key, folder, NORM_AVG_GL, NORM_STD_GL)
                yield self.fetch_data_x(key, folder, self.NORM_AVG_GL, self.NORM_STD_GL), self.fetch_data_y(f_dict, key)

    # @threadsafe_generator
    def pred_data_generator(self, test_flst, folder, batch_size):
        idx = 0
        total = len(test_flst)
        while idx < total:
            X = []
            for key in test_flst[idx:min(idx+batch_size, total)]:
                X.append(self.fetch_data_x(key, folder, self.NORM_AVG_TEST, self.NORM_STD_TEST))
            idx += batch_size
            yield np.array(X)

    @staticmethod
    def convert_y(y_, decode=True):
        temp = []
        if decode:
            for i, item in enumerate(y_):
                if item > 0.5:
                    temp.append(i)
        else:
            for i, item in enumerate(y_):
                if item > 0.5:
                    temp.append(1)
                else:
                    temp.append(0)

        return temp

    def save_to_file(self, fnames, predictions, decode=True, ckpt=None):

        resfile = 'result_' + ckpt + '.csv' if ckpt else 'result.csv'
        filepath = "./performance/" + resfile

        if decode:
            resfile = open(filepath, "w", newline="")
            res_writer = csv.writer(resfile)
            res_writer.writerow(["Id", "Predicted"])
            ans = dict()
            for fname, pred in zip(fnames, predictions):
                temp = self.convert_y(pred, decode=decode)
                ans[fname] = self.convert_res_str(temp)

            for key in sorted(ans.keys()):
                res_writer.writerow([key, ans[key]])

            resfile.close()
        else:
            ans = dict()
            for fname, pred in zip(fnames, predictions):
                ans[fname] = self.convert_y(pred, decode=decode)

            res = pd.DataFrame()
            for key in sorted(ans.keys()):
                res[key] = ans[key]

            res.to_csv(filepath, index=False)

    @staticmethod
    def f1_score(rr, pr):
        if rr == 0 and pr == 0:
            return 0
        else:
            return 2*rr*pr/(rr+pr)

    @staticmethod
    def df_to_dict(df):
        ans = dict()
        for key in df.keys():
            ans[key] = df[key].values

        return ans


