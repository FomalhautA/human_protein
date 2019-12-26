from util import *
from model import *
from loss import *

LR_DECAY = 0.9
TV_RATIO = 0.1

WP = np.array([0.5853, 0.9596, 0.8835, 0.9498, 0.9402, 0.9191, 0.9676, 0.9092, 0.9983, 0.9986, 0.9991, 0.9648, 0.9779,
               0.9827, 0.9657, 0.9993, 0.9829, 0.9932, 0.971, 0.9523, 0.9945, 0.8784, 0.9742, 0.9046, 0.9896, 0.7352,
               0.9894, 0.9996])
WN = np.array([0.4147, 0.0404, 0.1165, 0.0502, 0.0598, 0.0809, 0.0324, 0.0908, 0.0017, 0.0014, 0.0009, 0.0352, 0.0221,
               0.0173, 0.0343, 0.0007, 0.0171, 0.0068, 0.029, 0.0477, 0.0055, 0.1216, 0.0258, 0.0954, 0.0104, 0.2648,
               0.0106, 0.0004])

ORIG_CHANNEL = 4


def norm_params():
    f_dict_train, f_dict_val, f_dict_full = load_data()
    print("Full Set: ")
    channel_norm_params(f_dict_full.keys(), '../Data/train_s')
    print("Train Set: ")
    channel_norm_params(f_dict_train.keys(), '../Data/train_s')
    print("Validation Set: ")
    channel_norm_params(f_dict_val.keys(), '../Data/train_s')

    test_f_lst = os.listdir('../Data/test_s')
    test_f_set = set([item.split("_")[0] for item in test_f_lst])
    test_flst = list(test_f_set)
    channel_norm_params(test_flst, '../Data/test_s')


def train_input_fn(data_getter, batch_size):
    dataset = tf.data.Dataset.from_generator(lambda: data_getter,
                                             output_types=(tf.float32, tf.float32),
                                             output_shapes=(tf.TensorShape([None, None, 4]), tf.TensorShape([None, None, 28])),
                                             args=None)

    dataset = dataset.shuffle(buffer_size=10*batch_size)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    dataset = dataset.repeat(count=None)
    dataset = dataset.prefetch(buffer_size=1)

    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(data_getter, batch_size):
    dataset = tf.data.Dataset.from_generator(lambda: data_getter,
                                             output_types=(tf.float32, tf.float32),
                                             output_shapes=(tf.TensorShape([None, None, 4]), tf.TensorShape([None, None, 28])),
                                             args=None)

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    dataset = dataset.repeat(count=None)
    dataset = dataset.prefetch(buffer_size=1)

    return dataset.make_one_shot_iterator().get_next()


def pred_input_fn(data_getter, batch_size):
    dataset = tf.data.Dataset.from_generator(lambda: data_getter,
                                             output_types=tf.float32,
                                             output_shapes=(tf.TensorShape([None, None, 4])),
                                             args=None)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=1)

    return dataset.make_one_shot_iterator().get_next()


def model_fn(features, labels, mode, params):
    conv_params = arch_stone()
    logits = forward(features, conv_params, mode, fc1=1000, output=28)
    predictions = tf.cast(tf.math.greater(tf.squeeze(logits), 0.5), tf.int32)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions)

    else:
        loss = calculate_loss(tf.squeeze(logits),
                              tf.squeeze(labels),
                              mode='Focal',
                              weighted=True,
                              Wp=params['Wp'],
                              Wn=params['Wn'])

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.RMSPropOptimizer(learning_rate=params['lr'])
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step(), var_list=tf.trainable_variables())

            return tf.estimator.EstimatorSpec(mode,
                                              loss=loss,
                                              train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {'recall': tf.metrics.recall(tf.squeeze(labels), predictions),
                               'precision': tf.metrics.precision(tf.squeeze(labels), predictions),
                               'accuracy': tf.metrics.accuracy(tf.squeeze(labels), predictions)}
            return tf.estimator.EstimatorSpec(mode,
                                              loss=loss,
                                              eval_metric_ops=eval_metric_ops)
        else:
            raise Exception('Unsupported Mode Name {}'.format(mode))


def main_procedure():
    f_dict_train, f_dict_val, f_dict_full = load_data()
    test_fname = load_test_fname('../Data/test_s')
    scale_full = len(f_dict_full.keys())
    scale_train = len(f_dict_train.keys())
    scale_eval = len(f_dict_val.keys())
    print("full: {}, train: {}, eval: {}".format(scale_full, scale_train, scale_eval))

    epochs = 200
    train_batch_size = 32
    batch_cn_train = scale_train // train_batch_size + 1
    train_step = epochs * batch_cn_train
    eval_batch_size = 64
    eval_step = scale_eval // eval_batch_size + 1
    pred_batch_size = 64

    train_data_gen = train_data_generator(f_dict=f_dict_train, folder='../Data/train_s')
    eval_data_gen = eval_data_generator(f_dict=f_dict_val, folder='../Data/train_s')
    pred_data_gen = pred_data_generator(test_fname, folder='../Data/test_s')

    model_dir = './model'

    params = {'lr': 0.01,
              'Wp': WP.astype(np.float32),
              'Wn': WN.astype(np.float32)}

    config = tf.estimator.RunConfig(model_dir=model_dir,
                                    tf_random_seed=None,
                                    save_summary_steps=1,
                                    save_checkpoints_steps=10 * batch_cn_train,
                                    keep_checkpoint_max=80,
                                    log_step_count_steps=1)

    model = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir=model_dir,
                                   config=config,
                                   params=params,
                                   warm_start_from=None)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(train_data_gen,
                                                                        batch_size=train_batch_size),
                                        max_steps=train_step)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(eval_data_gen,
                                                                     batch_size=eval_batch_size),
                                      steps=eval_step,
                                      name=None,
                                      start_delay_secs=0,
                                      throttle_secs=0)

    x = tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

    print(x)



    # model.train(input_fn=lambda: train_input_fn(train_data_gen, batch_size=train_batch_size),
    #             steps=train_step)
    #
    # x = model.evaluate(input_fn=lambda: eval_input_fn(eval_data_gen,
    #                                                   batch_size=eval_batch_size),
    #                    steps=eval_step,
    #                    checkpoint_path='./model/model.ckpt-87400')
    #`
    # print(x)

    # predictions = model.predict(input_fn=lambda: pred_input_fn(pred_data_gen,
    #                                                            batch_size=pred_batch_size),
    #                             checkpoint_path='./model/model.ckpt-87400')
    #
    # save_to_file(test_fname, predictions)


if __name__ == '__main__':
    # data_partition(tv_ratio=0.1, ratio=1)
    # norm_params()
    main_procedure()

