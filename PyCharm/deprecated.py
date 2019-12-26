import copy
import time
import csv
import tensorflow as tf

from deprec_util import *


def train():
    f_dict_train, f_dict_val, f_dict_full = load_data(tv_ratio=TV_RATIO, ratio=1.)
    scale_full = len(f_dict_full.keys())
    train_scale = len(f_dict_train.keys())
    Y_full = fetch_batch_Y(f_dict_full, scale_full)

    X = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 4], name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, 28], name="Y")

    Wp = tf.placeholder(dtype=tf.float32, shape=[None, 28], name="Wp")
    Wn = tf.placeholder(dtype=tf.float32, shape=[None, 28], name="Wn")

    learning_rate = 0.001
    train_step = 4
    batch_scale = 64
    batch_cn = int(np.ceil(train_scale / batch_scale))
    val_step = 2
    saver_step = 2

    conv_params = arch_stone()
    logits = forward(X, conv_params, fc1=1000, output=28)

    # loss1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
    loss2 = tf.reduce_sum(calculate_loss(tf.squeeze(logits), tf.squeeze(Y), mode='Focal', weighted=True, Wp=Wp, Wn=Wn))

    # tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
    train_op = tf.contrib.layers.optimize_loss(loss2,
                                               global_step=tf.train.get_or_create_global_step(),
                                               learning_rate=learning_rate,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
                                               learning_rate_decay_fn=my_decay)

    cost_arr, costv = [], []
    acc_arr, accv = [], []
    rr_arr, rrv = [], []
    pr_arr, prv = [], []
    f1_arr, f1v = [], []

    print("train scale: ", train_scale)
    print('inner loop: ', batch_cn)

    W_p, W_n, N_p, N_n = get_weights(Y_full, scale=10)

    # X_batch, Y_batch = tf.train.shuffle_batch(tensors=[X_train_t, Y_train_t], batch_size=batch_scale, capacity=500,
    #                                           min_after_dequeue=150, allow_smaller_final_batch=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    saver = tf.train.Saver()
    # with tf.Session() as sess:
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for j in range(train_step):
        time_s = time.time()
        cost_count = 0
        f_dict_temp = copy.deepcopy(f_dict_train)
        for i in range(batch_cn):
            inputs_batch, labels_batch = batch_data_generator(f_dict_temp, batch_scale, folder='./Data/train_s')

            _, cost, y_ = sess.run([train_op, loss2, logits],
                                   feed_dict={X: inputs_batch, Y: labels_batch, Wp: W_p, Wn: W_n})

            sess.run([train_op], feed_dict={X: inputs_batch, Y: labels_batch, Wp: W_p, Wn: W_n})

            y_ = np.squeeze(y_)
            y_p = np.array(y_ > 0.5, dtype='int')

            y = labels_batch
            rr, pr, acc, f1 = evaluation(y_p, y)

            cost_count += cost
            cost_arr.append(cost)

            acc_arr.append(np.average(acc))
            rr_arr.append(np.average(rr))
            pr_arr.append(np.average(pr))
            f1_arr.append(np.average(f1))

        if j % val_step == 0:
            f_dict_temp = copy.deepcopy(f_dict_val)
            Y_, label = model_calc(sess, logits, X, f_dict_temp)
            rr, pr, acc, f1 = evaluation(Y_, label)

            accv.append(np.average(acc))
            rrv.append(np.average(rr))
            prv.append(np.average(pr))
            f1v.append(np.average(f1))

        if j % saver_step == saver_step-1:
            meta_flag = True if j == saver_step-1 else False
            saver.save(sess, "./model/inception_v3_"+str(j+1)+".ckpt", write_meta_graph=meta_flag)

        if j == 0:
            saver.save(sess, "./model/inception_v3_" + str(j + 1) + ".ckpt", write_meta_graph=True)

        time_cost = (time.time()-time_s)/60
        print("Step {}: {} Time Cost: {} min".format(j, round(cost_count / train_scale * 100, 4), round(time_cost, 1)))

    # train performance
    plot_performance(train_step, cost_arr, acc_arr, rr_arr, pr_arr, f1_arr, batch_cnt=batch_cn,
                     save_name="./performance/train_performance.jpg")

    # validation performance
    plot_performance(train_step, cost=None, acc=accv, rr=rrv, pr=prv, f1=f1v, step=val_step, batch_cnt=1,
                     save_name="./performance/validation_performance.jpg")

    # evaluation model performance in train set
    f_dict_temp = copy.deepcopy(f_dict_train)
    Y_, label = model_calc(sess, logits, X, f_dict_temp)
    res = evaluation(Y_, label, debug=True, write="./performance/validation_eval")

    # evaluation model performance in validation set
    f_dict_temp = copy.deepcopy(f_dict_val)
    Y_, label = model_calc(sess, logits, X, f_dict_temp)
    res = evaluation(Y_, label, debug=True, write="./performance/validation_eval")

    sess.close()


def batch_predict(sess, logits, X, flst):
    rows = []
    if len(flst) == 0:
        return rows

    image_arr = get_image_arr([fname for fname in flst], "../Data/test_s")
    normalizer(image_arr)
    image_arr = matrix_reshape(image_arr)

    y_test_ = sess.run(logits, feed_dict={X: image_arr})
    y_test_ = list(np.squeeze(y_test_))

    if len(flst) == 1:
        y_test_ = [y_test_]

    for fname, y_ in zip(flst, y_test_):
        predict = convert_y(y_)
        rows.append([fname, predict])

    return rows


def op_predict(sess, logits, X, flst):
    rows = []

    calc_batch = 128    # This should not be equal or larger than one
    total = len(flst)

    batch_cn = total // calc_batch

    for i in range(batch_cn):
        flst_batch = flst[i * calc_batch: (i + 1) * calc_batch]
        rows.extend(batch_predict(sess, logits, X, flst_batch))

    flst_batch = flst[batch_cn*calc_batch:]
    rows.extend(batch_predict(sess, logits, X, flst_batch))

    return rows


def _predict(meta, chk):

    test_f_lst = os.listdir("../Data/test_s")
    test_f_set = set([item.split("_")[0] for item in test_f_lst])
    test_flst = list(test_f_set)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta)
        # saver.restore(sess, tf.train.latest_checkpoint("./model"))
        saver.restore(sess, chk)

        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        logits = graph.get_tensor_by_name("logits:0")

        rows = op_predict(sess, logits, X, test_flst)

        save_to_file(rows)


def fully_connected(ap, params, activation=None):
    return tf.linalg.matmul(tf.squeeze(ap), params)


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
    pool1 = tf.keras.layers.MaxPooling2D(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    norm1 = batch_norm(pool1, scale1, beta1)

    #     norm1 = tf.nn.batch_normalization(pool1, depth_radius=5, bias=1, alpha=1, beta=0.5)

    conv2 = tf.nn.conv2d(norm1, filter2, strides=[1, 2, 2, 1], padding="VALID")
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 6, 6, 1], strides=[1, 1, 1, 1], padding='VALID')

    norm2 = batch_norm(pool2, scale2, beta2)

    #     norm2 = tf.nn.batch_normalization(pool2, depth_radius=5, bias=1, alpha=1, beta=0.5)

    local3 = tf.keras.layers.Dense(norm2, units=600, activation=tf.nn.sigmoid, use_bias=True)

    local4 = tf.keras.layers.Dense(local3, units=28, activation=tf.nn.sigmoid, use_bias=True)

    return local4


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

    # conv_params["c_Lfc_w"] = tf.Variable(np.random.randn(1024, 28), dtype=tf.float32)

    conv_params["c_Lfc_scale"] = tf.Variable(tf.ones([1, 1, 1000]), dtype=tf.float32)
    conv_params["c_Lfc_beta"] = tf.Variable(tf.zeros([1, 1, 1000]), dtype=tf.float32)

    conv_params["c_Lout_scale"] = tf.Variable(tf.ones([1, 1, 28]), dtype=tf.float32)
    conv_params["c_Lout_beta"] = tf.Variable(tf.zeros([1, 1, 28]), dtype=tf.float32)

    return conv_params


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


def model_calc(sess, logits, X, f_dict):
    calc_batch = 64
    inner_loop = len(f_dict.keys()) // calc_batch

    y_p = []
    labels = []
    for i in range(inner_loop):
        X_data, Y_data = batch_data_generator(f_dict, calc_batch, folder='./Data/train_s')
        y_ = sess.run(logits, feed_dict={X: X_data})
        y_ = np.squeeze(y_)
        y_p.append(np.array(y_ > 0.5, dtype='int'))
        labels.append(Y_data)

    if inner_loop * calc_batch < len(f_dict.keys()):
        X_data, Y_data = batch_data_generator(f_dict, calc_batch, folder='./Data/train_s')
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


def my_model_fn(features, labels, mode, params):
    conv_params = arch_stone()
    logits = forward(features, conv_params, mode, fc1=1000, output=28)

    loss = calculate_loss(tf.squeeze(logits),
                          tf.squeeze(labels),
                          mode='Focal',
                          weighted=True,
                          Wp=params['Wp'],
                          Wn=params['Wn'])
    # loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.squeeze(labels), tf.squeeze(logits)))

    predictions = tf.sign(tf.squeeze(logits) - 0.5)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=params['lr'])

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    eval_metric_ops = {'recall': tf.metrics.recall(tf.squeeze(labels), predictions),
                       'precision': tf.metrics.precision(tf.squeeze(labels), predictions),
                       'accuracy': tf.metrics.accuracy(tf.squeeze(labels), predictions)}

    return tf.estimator.EstimatorSpec(mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


def save_to_file(rows):
    resfile = open("./performance/result.csv", "w", newline="")
    res_writer = csv.writer(resfile)
    res_writer.writerow(["Id", "Predicted"])

    for row in rows:
        res_writer.writerow(row)

    resfile.close()


def _evaluate(meta='./model/model.ckpt-4360.meta', chk='./model/model.ckpt-4360'):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta)
        # saver.restore(sess, tf.train.latest_checkpoint("./model"))
        saver.restore(sess, chk)

        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        logits = graph.get_tensor_by_name("logits:0")

        rows = op_predict(sess, logits, X, test_flst)

        save_to_file(rows)


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
