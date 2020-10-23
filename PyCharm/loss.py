import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops


def calculate_loss(predict, labels, mode='Focal', weighted=False, Wp=None, Wn=None):
    if mode == 'CE':
        return loss_CE(predict, labels, weighted=weighted, Wp=Wp, Wn=Wn)

    elif mode == 'Focal':
        return loss_Focal(predict, labels, Wp=Wp, Wn=Wn)

    else:
        raise Exception('Unknown loss Mode.')


def loss_Focal(labels, predict, Wp=None, Wn=None, gamma=3, epsilon=1e-10):
    """
    Calculate Focal loss.
    predict, predict vector.
    labels, label vector.
    weighted, flag for weighted loss
    Wp, positive weight vector.
    Wn, negative weight vector.
    Return weighted loss.
    """
    if Wp is None or Wn is None:
        raise Exception('Focal loss need weight vectors!')

    p = tf.negative(tf.multiply(Wp,
                                tf.multiply(labels,
                                            tf.multiply(tf.pow(1-predict, gamma),
                                                        tf.math.log(predict + epsilon)))))

    n = tf.negative(tf.multiply(Wn,
                                tf.multiply((1 - labels),
                                            tf.multiply(tf.pow(predict, gamma),
                                                        tf.math.log(1 - predict + epsilon)))))

    loss = tf.add(p, n)

    # print('Loss is : ', loss.eval())

    return tf.reduce_mean(loss)


def focal_loss(y_true, y_pred, weights=None, gamma=3):
    """
    Compute focal loss for predictions.

    Multi-labels Focal loss formula:
        FL = -alpha*(z-p)^gamma * log(p) - (1-alpha)*p^gamma * log(1-p),
        where alpha=0.25, gamma=2, p = sigmoid(x), z=target_tensor

    :param y_true: a float tensor of shape [batch_size, num_labels] representing
                    one-hot encoded classification targets
    :param y_pred: a float tensor of shape [batch_size, num_labels] representing the predicted logits for each label.
    :param weights:
    :param gamma: a scalar tensor for focal loss gamma hyper-parameter

    :return: loss: a scalar tensor representing the value of the loss function
    """

    alpha = np.array(
        [0.5853, 0.9596, 0.8835, 0.9498, 0.9402, 0.9191, 0.9676, 0.9092, 0.9983, 0.9986, 0.9991, 0.9648, 0.9779,
         0.9827, 0.9657, 0.9993, 0.9829, 0.9932, 0.971, 0.9523, 0.9945, 0.8784, 0.9742, 0.9046, 0.9896, 0.7352,
         0.9894, 0.9996])
    # alpha = np.clip(alpha, 0.1, 0.9)
    # sigmoid_p = tf.nn.sigmoid(y_pred)
    sigmoid_p = y_pred
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For positive prediction, only need consider front part loss, back part is 0;
    # y_true > zeros <=> z=1, so positive coefficient = z - p

    pos_p_sub = array_ops.where(y_true > zeros, y_true - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0
    # y_true > zeros <=> z = 1, so negative coefficient = 0

    neg_p_sub = array_ops.where(y_true > zeros, zeros, sigmoid_p)

    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.math.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.)) \
                          - (1-alpha) * (neg_p_sub ** gamma) * tf.math.log(tf.clip_by_value(1.-sigmoid_p, 1e-8, 1.))

    return tf.reduce_sum(per_entry_cross_ent)


def f1_loss(y_true, y_pred):
    """
    Compute  f1-loss for predictions.

    :param y_true: a float tensor of shape [batch_size, num_labels] representing
                    one-hot encoded classification targets
    :param y_pred: a float tensor of shape [batch_size, num_labels] representing the predicted logits for each label.

    :return: loss: a scalar tensor representing the value of the loss function
    """
    epsilon = tf.keras.backend.epsilon()
    tp = tf.reduce_sum(tf.cast(y_true*y_pred, tf.dtypes.float32), axis=0)
    tn = tf.reduce_sum(tf.cast((1-y_true)*(1-y_pred), tf.dtypes.float32), axis=0)
    fp = tf.reduce_sum(tf.cast((1-y_true)*y_pred, tf.dtypes.float32), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true*(1-y_pred), tf.dtypes.float32), axis=0)

    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)

    f1 = 2*p*r / (p + r + epsilon)
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    return 1 - tf.reduce_mean(f1)


def loss_CE(predict, labels, weighted=False, Wp=None, Wn=None):
    """
    Calculate weighted loss.
    predict, predict vector.
    labels, label vector.
    weighted, flag for weighted loss
    Wp, positive weight vector.
    Wn, negative weight vector.
    Return weighted loss.
    """
    one = tf.constant(1., shape=[1, 28], dtype='float32')

    if weighted:
        p = tf.negative(tf.multiply(Wp, tf.multiply(labels, tf.math.log(predict))))
        n = tf.negative(tf.multiply(Wn, tf.multiply((1 - labels), tf.math.log(1 - predict))))
    else:
        p = tf.negative(tf.multiply(labels, tf.math.log(predict)))
        n = tf.negative(tf.multiply(tf.subtract(one, labels),
                                    tf.math.log(tf.subtract(one, predict))))

    loss = tf.add(p, n)

    return loss


