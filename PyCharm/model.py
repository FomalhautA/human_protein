import csv
import numpy as np
import tensorflow as tf


def batch_norm(tensor, scale, beta, epsilon=1e-10):
    """
    Batch Normalization.
    tensor, n dimensions tensor
    scale, n-1 dimensions tensor
    beta, n-1 dimensions tensor
    epsilon, small number to avoid dividing by zero
    """
    batch_mean, batch_var = tf.nn.moments(tensor, [0])

    return tf.nn.batch_normalization(x=tensor, mean=batch_mean, variance=batch_var, scale=scale, offset=beta,
                                     variance_epsilon=epsilon)


def inception_init(inputs, out1, r3, out3, r5, out5, m, s1, s2):
    res = dict()

    out = out1 + out3 + out5 + m

    res["c_1_1"] = tf.Variable(np.random.randn(1, 1, inputs, out1), dtype=tf.float32)
    res["c_r3_1_1"] = tf.Variable(np.random.randn(1, 1, inputs, r3), dtype=tf.float32)
    res["c_3_3"] = tf.Variable(np.random.randn(3, 3, r3, out3), dtype=tf.float32)
    res["c_r5_1_1"] = tf.Variable(np.random.randn(1, 1, inputs, r5), dtype=tf.float32)
    res["c_5_5_1"] = tf.Variable(np.random.randn(3, 3, r5, out5), dtype=tf.float32)
    res["c_5_5_2"] = tf.Variable(np.random.randn(3, 3, out5, out5), dtype=tf.float32)
    res["c_m_1_1"] = tf.Variable(np.random.randn(1, 1, inputs, m), dtype=tf.float32)

    res["scale"] = tf.Variable(tf.ones([s1, s2, out]), dtype=tf.float32)
    res["beta"] = tf.Variable(tf.zeros([s1, s2, out]), dtype=tf.float32)

    return res


def arch_stone():
    conv_params = dict()

    conv_params["c_L1_7_7"] = tf.Variable(np.random.randn(7, 7, 4, 64), dtype=tf.float32)
    conv_params["c_L1_scale"] = tf.Variable(tf.ones([128, 128, 64]), dtype=tf.float32)
    conv_params["c_L1_beta"] = tf.Variable(tf.zeros([128, 128, 64]), dtype=tf.float32)

    conv_params["c_L2_1_1"] = tf.Variable(np.random.randn(1, 1, 64, 64), dtype=tf.float32)
    conv_params["c_L2_3_3"] = tf.Variable(np.random.randn(3, 3, 64, 192), dtype=tf.float32)

    conv_params["c_L2_scale"] = tf.Variable(tf.ones([64, 64, 192]), dtype=tf.float32)
    conv_params["c_L2_beta"] = tf.Variable(tf.zeros([64, 64, 192]), dtype=tf.float32)

    conv_params["c_L3a"] = inception_init(inputs=192, out1=64, r3=96, out3=128, r5=16, out5=32, m=32, s1=32, s2=32)

    conv_params["c_L3b"] = inception_init(inputs=256, out1=128, r3=128, out3=192, r5=32, out5=96, m=64, s1=32, s2=32)

    conv_params["c_L4a"] = inception_init(inputs=480, out1=192, r3=96, out3=208, r5=16, out5=48, m=64, s1=16, s2=16)

    conv_params["c_L4b"] = inception_init(inputs=512, out1=160, r3=112, out3=224, r5=24, out5=64, m=64, s1=16, s2=16)

    conv_params["c_L4c"] = inception_init(inputs=512, out1=128, r3=128, out3=256, r5=24, out5=64, m=64, s1=16, s2=16)

    conv_params["c_L4d"] = inception_init(inputs=512, out1=112, r3=144, out3=288, r5=32, out5=64, m=64, s1=16, s2=16)

    conv_params["c_L4e"] = inception_init(inputs=528, out1=256, r3=160, out3=320, r5=32, out5=128, m=128, s1=16, s2=16)

    conv_params["c_L5a"] = inception_init(inputs=832, out1=256, r3=160, out3=320, r5=32, out5=128, m=128, s1=8, s2=8)

    conv_params["c_L5b"] = inception_init(inputs=832, out1=384, r3=192, out3=384, r5=48, out5=128, m=128, s1=8, s2=8)

    conv_params["c_Lfc_scale"] = tf.Variable(tf.ones([1, 1, 1000]), dtype=tf.float32)
    conv_params["c_Lfc_beta"] = tf.Variable(tf.zeros([1, 1, 1000]), dtype=tf.float32)

    conv_params["c_Lout_scale"] = tf.Variable(tf.ones([1, 1, 28]), dtype=tf.float32)
    conv_params["c_Lout_beta"] = tf.Variable(tf.zeros([1, 1, 28]), dtype=tf.float32)

    return conv_params


def inception(inputs, params):
    c_1_1 = params["c_1_1"]
    c_r3_1_1 = params["c_r3_1_1"]
    c_3_3 = params["c_3_3"]
    c_r5_1_1 = params["c_r5_1_1"]
    c_5_5_1 = params["c_5_5_1"]
    c_5_5_2 = params["c_5_5_2"]
    c_m_1_1 = params["c_m_1_1"]
    scale = params["scale"]
    beta = params["beta"]

    z1 = tf.nn.conv2d(inputs, c_1_1, strides=[1, 1, 1, 1], padding="SAME")

    z_r3 = tf.nn.conv2d(inputs, c_r3_1_1, strides=[1, 1, 1, 1], padding="SAME")
    a_r3 = tf.nn.leaky_relu(z_r3)
    z3 = tf.nn.conv2d(a_r3, c_3_3, strides=[1, 1, 1, 1], padding="SAME")

    z_r5 = tf.nn.conv2d(inputs, c_r5_1_1, strides=[1, 1, 1, 1], padding="SAME")
    a_r5 = tf.nn.leaky_relu(z_r5)
    z5_1 = tf.nn.conv2d(a_r5, c_5_5_1, strides=[1, 1, 1, 1], padding="SAME")
    a5_1 = tf.nn.leaky_relu(z5_1)
    z5_2 = tf.nn.conv2d(a5_1, c_5_5_2, strides=[1, 1, 1, 1], padding="SAME")

    m = tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME")
    zm = tf.nn.conv2d(m, c_m_1_1, strides=[1, 1, 1, 1], padding="SAME")

    z = tf.concat(values=[z1, z3, z5_2, zm], axis=3)
    # bn = tf.layers.batch_normalization(z, axis=-1)
    bn = batch_norm(z, scale, beta)
    a = tf.nn.leaky_relu(bn)

    return a


def forward(X, params, fc1=1000, output=28):
    # Layer 1
    z1 = tf.nn.conv2d(input=X, filter=params["c_L1_7_7"], strides=[1, 2, 2, 1], padding="SAME")
    bn1 = batch_norm(z1, params["c_L1_scale"], params["c_L1_beta"])
    a1 = tf.nn.leaky_relu(bn1)
    m1 = tf.nn.max_pool(a1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Layer 2
    z_2r = tf.nn.conv2d(input=m1, filter=params["c_L2_1_1"], strides=[1, 1, 1, 1], padding="VALID")
    a_2r = tf.nn.leaky_relu(z_2r)
    # ToDo: Maybe need batch norm.
    z2 = tf.nn.conv2d(input=z_2r, filter=params["c_L2_3_3"], strides=[1, 1, 1, 1], padding="SAME")
    bn2 = batch_norm(z2, params["c_L2_scale"], params["c_L2_beta"])
    a2 = tf.nn.leaky_relu(bn2)

    # Max Pool
    m2 = tf.nn.max_pool(a2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Layer 3a
    a_3a = inception(m2, params["c_L3a"])
    # layer 3b
    a_3b = inception(a_3a, params["c_L3b"])

    # Max Pool
    m3 = tf.nn.max_pool(a_3b, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Layer 4a
    a_4a = inception(m3, params["c_L4a"])

    # Layer 4b
    a_4b = inception(a_4a, params["c_L4b"])
    # Layer 4c
    a_4c = inception(a_4b, params["c_L4c"])
    # Layer 4d
    a_4d = inception(a_4c, params["c_L4d"])
    # Layer 4e
    a_4e = inception(a_4d, params["c_L4e"])

    # Max Pool
    m4 = tf.nn.max_pool(a_4e, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Layer 5a
    a_5a = inception(m4, params["c_L5a"])
    # Layer 5b
    a_5b = inception(a_5a, params["c_L5b"])

    # Avg Pool
    ap = tf.nn.avg_pool(a_5b, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding="VALID")

    # Full Connect
    z_fc = tf.layers.dense(ap, units=output, activation=None)
    # bnfc = tf.layers.batch_normalization(z_fc, axis=)
    bnfc = batch_norm(z_fc, params["c_Lout_scale"], params["c_Lout_beta"])
    a_out = tf.nn.sigmoid(bnfc, name="logits")

    #     # Full Connect
    #     z_out = tf.contrib.layers.fully_connected(a_fc, num_outputs=output, activation_fn=None)
    # #     bnout = batch_norm(z_out, params["c_Lout_scale"], params["c_Lout_beta"], epsilon)
    #     a_out = tf.nn.sigmoid(z_out)

    return a_out


def inference(inputs, filter1, filter2, scale1, beta1, scale2, beta2):
    """
    Forward calculation.

    inputs, input tensor
    filter1,
    filter2,
    scale1,
    beta1,
    scale2,
    beta2,
    epsilon,

    Return logits vectors.
    """

    conv1 = tf.nn.conv2d(inputs, filter1, strides=[1, 8, 8, 1], padding="VALID")
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    norm1 = batch_norm(pool1, scale1, beta1)

    #     norm1 = tf.nn.batch_normalization(pool1, depth_radius=5, bias=1, alpha=1, beta=0.5)

    conv2 = tf.nn.conv2d(norm1, filter2, strides=[1, 2, 2, 1], padding="VALID")
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 6, 6, 1], strides=[1, 1, 1, 1], padding='VALID')

    norm2 = batch_norm(pool2, scale2, beta2)

    #     norm2 = tf.nn.batch_normalization(pool2, depth_radius=5, bias=1, alpha=1, beta=0.5)

    local3 = tf.layers.dense(norm2, units=600, activation=tf.nn.sigmoid, use_bias=True)

    local4 = tf.layers.dense(local3, units=28, activation=tf.nn.sigmoid, use_bias=True)

    return local4


def calculate_loss(predict, labels, mode='Focal', weighted=False, Wp=None, Wn=None):
    if mode == 'CE':
        return loss_CE(predict, labels, weighted=weighted, Wp=Wp, Wn=Wn)

    elif mode == 'Focal':
        return loss_Focal(predict, labels, Wp=Wp, Wn=Wn)

    else:
        raise Exception('Unknown loss Mode.')


def loss_Focal(predict, labels, Wp=None, Wn=None, gamma=2, epsilon=1e-10):
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

    p = tf.negative(tf.multiply(tf.convert_to_tensor(Wp, tf.float32),
                                tf.multiply(labels,
                                            tf.multiply(tf.pow((1-predict), gamma),
                                                        tf.log(predict + epsilon)))))
    n = tf.negative(tf.multiply(tf.convert_to_tensor(Wn, tf.float32),
                                tf.multiply((1 - labels),
                                            tf.multiply(tf.pow(predict, gamma),
                                                        tf.log(1 - predict + epsilon)))))

    loss = tf.add(p, n)

    # print('Loss is : ', loss.eval())

    return loss


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
        p = tf.negative(tf.multiply(Wp, tf.multiply(labels, tf.log(predict))))
        n = tf.negative(tf.multiply(Wn, tf.multiply((1 - labels), tf.log(1 - predict))))
    else:
        p = tf.negative(tf.multiply(labels, tf.log(predict)))
        n = tf.negative(tf.multiply(tf.subtract(one, labels),
                                    tf.log(tf.subtract(one, predict))))

    loss = tf.add(p, n)

    return loss


def calculate_rate(predict, label, p, l):
    """
    Count for four different types: 10, 01, 00, 11.
    """

    stat = np.array([[1 if p_item == p and l_item == l else 0
                      for p_item, l_item in zip(p_, l_)]
                     for p_, l_ in zip(predict.T, label.T)])

    rate = np.sum(stat.T, axis=0)

    return rate


def evaluation(y_, y, debug=False, write=None):
    """
    Calculate tp, fp, tn, fn.
    Calculate rr, pr, acc, f1-score.
    Return rr, pr, acc, f1-score.
    y_: predictions (logits)
    y: labels
    write: filename for recording evaluation result
    """
    scale = len(y)  # sample scale
    y = np.squeeze(y)
    y = np.array(y, dtype='int')

    tn = calculate_rate(y_, y, p=0, l=0)
    tp = calculate_rate(y_, y, p=1, l=1)
    fn = calculate_rate(y_, y, p=0, l=1)
    fp = calculate_rate(y_, y, p=1, l=0)

    rr = [0 if (tp_item + fn_item) == 0 else tp_item /
                                             (tp_item + fn_item) for tp_item, fn_item in zip(tp, fn)]
    pr = [0 if (tp_item + fp_item) == 0 else tp_item /
                                             (tp_item + fp_item) for tp_item, fp_item in zip(tp, fp)]
    acc = (tp + tn) / scale

    f1 = [0 if (pr_item + rr_item == 0) else
          (2 * pr_item * rr_item) / (pr_item + rr_item) for pr_item, rr_item in zip(pr, rr)]

    rr_avg = np.average(rr)
    pr_avg = np.average(pr)
    acc_avg = np.average(acc)
    f1_avg = np.average(f1)

    if debug:
        print("class, ( tp, fp, tn, fn)")
        for i in range(len(tp)):
            print(i, " : ", (tp[i], fp[i], tn[i], fn[i]))

        #         print("tp: ", tp)
        #         print("fn: ", fn)
        #         print("tn: ", tn)
        #         print("fp: ", fp)

        print("class, ( rr, pr, acc, f1)")
        for i in range(len(rr)):
            print(i, " : ", (round(rr[i], 2), round(pr[i], 2), round(acc[i], 2),
                             round(f1[i], 2)))

        #         print("rr: ", [round(item, 3) for item in rr])
        #         print("pr: ", [round(item, 3) for item in pr])
        #         print("acc: ", [round(item, 3) for item in acc])
        #         print("f1: ", [round(item, 3) for item in f1])

        print("accuracy : ", round(acc_avg, 4))
        print(" rr : ", round(rr_avg, 4))
        print(" pr : ", round(pr_avg, 4))
        print(" f1 : ", round(f1_avg, 4))

    if write:
        resfile = open(write+"_count.csv", "w", newline="")
        res_writer = csv.writer(resfile)
        res_writer.writerow(["Class", "rr", "pr", "acc", "f1"])
        for i in range(len(tp)):
            res_writer.writerow([i, tp[i], fp[i], tn[i], fn[i]])

        resfile.close()

        resfile = open(write + "_index.csv", "w", newline="")
        res_writer = csv.writer(resfile)
        res_writer.writerow(["Class", "tp", "fp", "tn", "fn"])
        for i in range(len(rr)):
            res_writer.writerow([i, round(rr[i], 2), round(pr[i], 2), round(acc[i], 2), round(f1[i], 2)])

        resfile.close()

    return rr, pr, acc, f1
