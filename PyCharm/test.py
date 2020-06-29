import tensorflow as tf
from util import f1_score, get_image


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())

    print('FLOPs: {};  Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))


def calc_f1score():
    prs = [0.529, 0.5892]
    rrs = [0.6609, 0.6548]
    idx = [200, 800]

    for (i, pr, rr) in zip(idx, prs, rrs):
        print(i, pr, rr, round(f1_score(pr, rr), 4))


def stat_model():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./model/model.ckpt-0.meta')
        saver.restore(sess, './model/model.ckpt-0')

        graph = tf.get_default_graph()

        stats_graph(graph)


def _test():
    a = get_image('000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0', '../Data/train')
    b = a
    print(b.shape)
    # b.astype('uint8')
    # img = Image.fromarray(b)
    # img.show()

    # im = Image.open(os.path.join('../Data/test_s', '00ad3e84-bad1-11e8-b2b8-ac1f6b6435d0' + '_' + 'green' + '.png'), 'r')
    # a = np.array(im.convert('RGB'))

    # f_dict_train, f_dict_val, f_dict_full = load_data(tv_ratio=TV_RATIO, ratio=1.)
    #
    # a = train_data_generator(f_dict=f_dict_train, folder='../Data/train_s')
    # b, c = next(a)
    # print(b.shape)
    # # print(b)
    # print(c.shape)
    # print(c)

    # Y_full = fetch_batch_Y(f_dict_full, scale_full)
    # Wp, Wn, Np, Nn = get_weights(Y_full)
    #
    # print('Positive Weight: ', [round(item, 4) for item in Wp.tolist()])
    # print('Negative weight: ', [round(item, 4) for item in Wn.tolist()])

    # features, labels = input_fn(mode=tf.estimator.ModeKeys.TRAIN, batch_size=2)
    #
    # with tf.Session() as sess:
    #
    #     # loss = loss_Focal(predict, label, Wp=WP.astype(np.float32), Wn=WN.astype(np.float32))
    #     x = sess.run(features)
    #     y = sess.run(labels)
    #
    #     print(x.shape)
    #     print(y.shape)

    # calc_f1score()


if __name__ == '__main__':
    _test()
