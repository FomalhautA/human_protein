import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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


def image_input_fn(f_dict, mode, batch_size=128):
    if mode == tf.estimator.ModeKeys.TRAIN:
        features, labels = batch_data_generator(f_dict,
                                                batch_size,
                                                folder='../Data/train_s',
                                                replacement=False)

    elif mode == tf.estimator.ModeKeys.EVAL:
        features, labels = batch_data_generator(f_dict,
                                                batch_size,
                                                folder='../Data/train_s',
                                                replacement=False)

    elif mode == tf.estimator.ModeKeys.PREDICT:
        features, labels = batch_data_generator(f_dict,
                                                batch_size,
                                                folder='../Data/test_s',
                                                replacement=False, labels=False)
    else:
        raise Exception('Invalid Mode for Estimator!')

    return features, labels


def batch_data_generator(f_dict, batch_size, folder, replacement=False, labels=True):
    """

    :param f_dict:
    :param batch_size:
    :param folder
    :param replacement: True, reset; False, abandon;
    :param labels
    :return:
    """
    print('Current data counts: ', len(f_dict.keys()))
    if len(f_dict.keys()) < batch_size:
        raise Exception(StopIteration)

    scale = min(len(f_dict.keys()), batch_size)

    batch_dict = rand_sample(f_dict, scale)
    if not replacement:
        dict_substract(f_dict, batch_dict)

    res = fetch_batch_data(batch_dict, scale, 0, folder, labels=labels)

    if labels:
        return tf.convert_to_tensor(res[0], tf.float32), tf.convert_to_tensor(res[1], tf.float32)

    else:
        return tf.convert_to_tensor(res[0], tf.float32), None


def fetch_batch_data(f_dict, scale, idx, folder, labels=True):
    """
    f_dict, filename dictionary
    scale, batch size
    idx, batch number

    """
    X = fetch_batch_X(f_dict, scale, idx, folder)
    X = normalizer_batch(X)
    X = matrix_reshape(X)

    if labels:
        Y = fetch_batch_Y(f_dict, scale, idx)
        return X, Y

    else:
        return X


def fetch_batch_Y(f_dict, scale, idx=0):
    pick_list = list(f_dict.keys())

    fname_batch_ = pick_list[idx * scale: (idx + 1) * scale]
    # fname_green_batch_ = [item + "_green.png" for item in fname_batch_]

    labels_batch_ = pick_labels(f_dict, fname_batch_)
    labels = labels_batch_
    labels = labels.reshape((-1, 1, 1, 28))

    return labels


def fetch_batch_X(f_dict, scale, idx, folder):
    pick_list = list(f_dict.keys())

    fname_batch_ = pick_list[idx * scale: (idx + 1) * scale]
    # fname_green_batch_ = [item + "_green.png" for item in fname_batch_]

    image_arr = get_image_arr(fname_batch_, folder)

    return image_arr


def normalizer_batch(mat):
    temp = copy.deepcopy(mat)
    temp = temp.astype(float)
    for i in range(ORIG_CHANNEL):
        temp[:, i, :, :] = (temp[:, i, :, :] - NORM_AVG_GL[i]) / NORM_STD_GL[i]

    return temp


def my_decay(a, b):
    """
    Exponential decay function with decay_rate, decay_steps.
    """
    return tf.train.exponential_decay(a, b, decay_steps=2, decay_rate=LR_DECAY, staircase=True)


def matrix_reshape(mat):
    temp = []
    for i in range(mat.shape[0]):
        temp.append(image_reshape(mat[i]))
    return np.array(temp)


def image_reshape(mat):
    c, h, w = mat.shape
    temp = np.zeros((h, w, c))
    for i in range(h):
        for j in range(w):
            for k in range(c):
                temp[i][j][k] = mat[k][i][j]

    return temp


def batch_norm(tensor, scale, beta, epsilon=1e-10):
    """
    Batch Normalization.
    tensor, n dimensions tensor
    scale, n-1 dimensions tensor
    beta, n-1 dimensions tensor
    epsilon, small number to avoid dividing by zero
    """
    batch_mean, batch_var = tf.nn.moments(tensor, [0, 1, 2])

    return tf.nn.batch_normalization(x=tensor, mean=batch_mean, variance=batch_var, scale=scale, offset=beta,
                                     variance_epsilon=epsilon)


