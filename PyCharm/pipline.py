import time

from util import *
from model import *
from tensorflow_estimator import estimator


LR_DECAY = 0.9
TV_RATIO = 0.1

NORM_AVG = [20.58, 13.46, 14., 21.12]
NORM_STD = [15.82, 13.45, 25.99, 16.03]

ORIG_CHANNEL = 4


def norm_params():
    print("Train set ratio: {}".format(1-TV_RATIO))
    f_dict_train, f_dict_val, f_dict_full = load_data(tv_ratio=TV_RATIO, ratio=1.)
    print("Full Set: ")
    channel_norm_params(f_dict_full.keys(), '../Data/train_s')
    print("Train Set: ")
    channel_norm_params(f_dict_train.keys(), '../Data/train_s')
    print("Validation Set: ")
    channel_norm_params(f_dict_val.keys(), '../Data/train_s')


def my_model_fn()


def main_procedure():
    est = estimator.Estimator(model_fn, model_dir=None, config=None, params=None, warm_start_from=None)






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
    logits = model(X, conv_params, fc1=1000, output=28)

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
        # time_s = time.time()
        # cost_count = 0
        f_dict_temp = copy.deepcopy(f_dict_train)
        for i in range(batch_cn):
            inputs_batch, labels_batch = batch_data_generator(f_dict_temp, batch_scale)

            # _, cost, y_ = sess.run([train_op, loss2, logits],
            #                        feed_dict={X: inputs_batch, Y: labels_batch, Wp: W_p, Wn: W_n})

            sess.run([train_op], feed_dict={X: inputs_batch, Y: labels_batch, Wp: W_p, Wn: W_n})

            # y_ = np.squeeze(y_)
            # y_p = np.array(y_ > 0.5, dtype='int')
            #
            # y = labels_batch
            # rr, pr, acc, f1 = evaluation(y_p, y)
            #
            # cost_count += cost
            # cost_arr.append(cost)
            #
            # acc_arr.append(np.average(acc))
            # rr_arr.append(np.average(rr))
            # pr_arr.append(np.average(pr))
            # f1_arr.append(np.average(f1))

        # if j % val_step == 0:
        #     f_dict_temp = copy.deepcopy(f_dict_val)
        #     Y_, label = model_calc(sess, logits, X, f_dict_temp)
        #     rr, pr, acc, f1 = evaluation(Y_, label)
        #
        #     accv.append(np.average(acc))
        #     rrv.append(np.average(rr))
        #     prv.append(np.average(pr))
        #     f1v.append(np.average(f1))
        #
        # if j % saver_step == saver_step-1:
        #     meta_flag = True if j == saver_step-1 else False
        #     saver.save(sess, "./model/inception_v3_"+str(j+1)+".ckpt", write_meta_graph=meta_flag)
        #
        # if j == 0:
        #     saver.save(sess, "./model/inception_v3_" + str(j + 1) + ".ckpt", write_meta_graph=True)
        #
        # time_cost = (time.time()-time_s)/60
        # print("Step {}: {} Time Cost: {} min".format(j, round(cost_count / train_scale * 100, 4), round(time_cost, 1)))

    # # train performance
    # plot_performance(train_step, cost_arr, acc_arr, rr_arr, pr_arr, f1_arr, batch_cnt=batch_cn,
    #                  save_name="./performance/train_performance.jpg")
    #
    # # validation performance
    # plot_performance(train_step, cost=None, acc=accv, rr=rrv, pr=prv, f1=f1v, step=val_step, batch_cnt=1,
    #                  save_name="./performance/validation_performance.jpg")

    # evaluation model performance in train set
    f_dict_temp = copy.deepcopy(f_dict_train)
    Y_, label = model_calc(sess, logits, X, f_dict_temp)
    res = evaluation(Y_, label, debug=True, write="./performance/validation_eval")

    # evaluation model performance in validation set
    f_dict_temp = copy.deepcopy(f_dict_val)
    Y_, label = model_calc(sess, logits, X, f_dict_temp)
    res = evaluation(Y_, label, debug=True, write="./performance/validation_eval")

    sess.close()


def convert_y(y_):
    temp = []
    for i, item in enumerate(y_):
        if item > 0.5:
            temp.append(i)

    predict = convert_res_str(temp)

    return predict


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


def save_to_file(rows):
    resfile = open("./performance/result.csv", "w", newline="")
    res_writer = csv.writer(resfile)
    res_writer.writerow(["Id", "Predicted"])

    for row in rows:
        res_writer.writerow(row)

    resfile.close()


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


def _test():
    # a = get_image_arr(['00a79920-bad1-11e8-b2b8-ac1f6b6435d0', '00ad3e84-bad1-11e8-b2b8-ac1f6b6435d0'], '../Data/test_s')
    # b = a[1][3]*255
    # print(b.shape)
    # b.astype('uint8')
    # img = Image.fromarray(b)
    # img.show()

    # im = Image.open(os.path.join('../Data/test_s', '00ad3e84-bad1-11e8-b2b8-ac1f6b6435d0' + '_' + 'green' + '.png'), 'r')
    # a = np.array(im.convert('RGB'))

    # a = np.random.rand(3, 4, 4)
    # print(a)
    # b = np.vstack((a[0], a[1]))
    # print(b.shape)
    # print(b)

    f_dict_train, f_dict_val, f_dict_full = load_data(tv_ratio=TV_RATIO, ratio=1.)
    scale_full = len(f_dict_full.keys())
    Y_full = fetch_batch_Y(f_dict_full, scale_full)

    W_p, W_n, N_p, N_n = get_weights(Y_full, scale=10)

    showWeights(W_p, W_n, N_p, N_n)


if __name__ == '__main__':
    # _test()
    train()
    # _predict("./model_9_96_70/inception_v3_1.ckpt.meta", "./model_9_96_70/inception_v3_40.ckpt")

