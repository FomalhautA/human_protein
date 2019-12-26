import tensorflow as tf


def calculate_loss(predict, labels, mode='Focal', weighted=False, Wp=None, Wn=None):
    if mode == 'CE':
        return loss_CE(predict, labels, weighted=weighted, Wp=Wp, Wn=Wn)

    elif mode == 'Focal':
        return loss_Focal(predict, labels, Wp=Wp, Wn=Wn)

    else:
        raise Exception('Unknown loss Mode.')


def loss_Focal(predict, labels, Wp=None, Wn=None, gamma=3, epsilon=1e-10):
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


