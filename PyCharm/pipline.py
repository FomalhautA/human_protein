import os
import pandas as pd

from util import *
from model import *
from loss import *
from callbacks import EarlyStoppingCallback
from resnext import create_model_resnext

LR_DECAY = 0.9
TV_RATIO = 0.1

WP = np.array([0.5853, 0.9596, 0.8835, 0.9498, 0.9402, 0.9191, 0.9676, 0.9092, 0.9983, 0.9986, 0.9991, 0.9648, 0.9779,
               0.9827, 0.9657, 0.9993, 0.9829, 0.9932, 0.971, 0.9523, 0.9945, 0.8784, 0.9742, 0.9046, 0.9896, 0.7352,
               0.9894, 0.9996])
WN = np.array([0.4147, 0.0404, 0.1165, 0.0502, 0.0598, 0.0809, 0.0324, 0.0908, 0.0017, 0.0014, 0.0009, 0.0352, 0.0221,
               0.0173, 0.0343, 0.0007, 0.0171, 0.0068, 0.029, 0.0477, 0.0055, 0.1216, 0.0258, 0.0954, 0.0104, 0.2648,
               0.0106, 0.0004])

# WP = np.array([0.6174, 0.9644, 0.8852, 0.9623, 0.9524, 0.9148, 0.9641, 0.9373, 0.966, 0.9629, 0.9633, 0.9639, 0.9653,
#                0.9629, 0.964, 0.9621, 0.9591, 0.9647, 0.9658, 0.9619, 0.965, 0.9, 0.9658, 0.9252, 0.9668, 0.7894,
#                0.9645, 0.9591])
# WN = np.array([0.3826, 0.0356, 0.1148, 0.0377, 0.0476, 0.0852, 0.0359, 0.0627, 0.034, 0.0371, 0.0367, 0.0361, 0.0347,
#                0.0371, 0.036, 0.0379, 0.0409, 0.0353, 0.0342, 0.0381, 0.035, 0.1, 0.0342, 0.0748, 0.0332, 0.2106,
#                0.0355, 0.0409])

ORIG_CHANNEL = 4


def norm_params(train='../Data/tra.csv', val='../Data/val.csv', full='../Data/full.csv'):
    data_utils = DataUtils()

    f_dict_train, f_dict_val, f_dict_full = data_utils.load_data(train=train, val=val, full=full)
    print("Full Set: ")
    data_utils.channel_norm_params(f_dict_full.keys(), '../Data/train_s')
    print("Train Set: ")
    data_utils.channel_norm_params(f_dict_train.keys(), '../Data/train_s')
    print("Validation Set: ")
    data_utils.channel_norm_params(f_dict_val.keys(), '../Data/train_s')

    test_f_lst = os.listdir('../Data/test_s')
    test_f_set = set([item.split("_")[0] for item in test_f_lst])
    test_flst = list(test_f_set)
    data_utils.channel_norm_params(test_flst, '../Data/test_s')


def train_input_fn(data_getter, batch_size):
    dataset = tf.data.Dataset.from_generator(lambda: data_getter,
                                             output_types=(tf.float32, tf.float32),
                                             output_shapes=(tf.TensorShape([None, None, 4]), tf.TensorShape([None, None, 28])),
                                             args=None)

    dataset = dataset.shuffle(buffer_size=8*batch_size)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    dataset = dataset.repeat(count=None)
    dataset = dataset.prefetch(buffer_size=4)

    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()


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


def train_pipline(train='../Data/train.csv', val='../Data/val.csv', full='../Data/full.csv', ckpt=None):
    data_utils = DataUtils()

    f_dict_train = data_utils.df_to_dict(pd.read_csv(train))
    f_dict_val = data_utils.df_to_dict(pd.read_csv(val))
    f_dict_full = data_utils.df_to_dict(pd.read_csv(full))

    scale_full = len(f_dict_full.keys())
    scale_train = len(f_dict_train.keys())
    scale_eval = len(f_dict_val.keys())
    print("samples count: {}, train: {}, val: {}".format(scale_full, scale_train, scale_eval))

    epochs = 100
    batch_size = 16
    validation_split = 0.2
    lr = 0.1
    lr_decay = 0.84
    decay_epoch = 5
    steps_per_epoch = int(np.ceil(scale_full / batch_size))
    save_epochs = 5
    decay_steps = decay_epoch * steps_per_epoch

    train_data_gen = data_utils.train_data_generator(f_dict=f_dict_full, folder='../Data/train_s', batch_size=batch_size)
    # eval_data_gen = eval_data_generator(f_dict=f_dict_val, folder='../Data/train_s')

    model_dir = './model'

    # conv_params = resnet_arch_stone()
    # conv_params = inception_arch_stone()

    if ckpt is not None:
        model = tf.keras.models.load_model(os.path.join(model_dir, ckpt),
                                           custom_objects={'leaky_relu': tf.nn.leaky_relu,
                                                           'focal_loss': focal_loss,
                                                           'f1_loss': f1_loss})
    else:
        model = create_model_resnext(lr=lr, batch_size=batch_size, lr_decay=lr_decay, decay_steps=decay_steps)
        # model = create_model_inception(conv_params, lr=lr, batch_size=batch_size, lr_decay=lr_decay,
        # decay_steps=decay_steps)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_dir, 'model_{epoch}.h5'),
                                                     save_weights_only=False, verbose=1, save_best_only=False,
                                                     save_freq=save_epochs*steps_per_epoch)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(model_dir, 'logs'), profile_batch=0,
                                                 write_graph=True, write_images=False, embeddings_freq=0,
                                                 histogram_freq=1, update_freq='epoch')

    # earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='binary_accuracy', min_delta=0.0001, patience=5,
    #                                                       verbose=0, mode='max')

    estop_callback = EarlyStoppingCallback()

    model.fit(train_data_gen, epochs=epochs,
              callbacks=[cp_callback, tb_callback, estop_callback], validation_data=None,
              initial_epoch=0, steps_per_epoch=steps_per_epoch, validation_steps=None, validation_batch_size=None,
              validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)

    model.save(filepath=model_dir, overwrite=True, include_optimizer=True, save_format='tf',
               signatures=None, options=None)

    model.summary()


def model_evaluation(model_file='./model/model_180.h5'):
    print('Model {} predict ...'.format(model_file.split('/')[-1]))

    data_utils = DataUtils()

    batch_size = 32
    test_fname = data_utils.load_test_fname('../Data/test_s')
    steps = np.ceil(len(test_fname) / batch_size)

    pred_data_gen = data_utils.pred_data_generator(test_fname, folder='../Data/test_s', batch_size=batch_size)
    model = tf.keras.models.load_model(model_file,
                                       custom_objects={'leaky_relu': tf.nn.leaky_relu,
                                                       'focal_loss': focal_loss,
                                                       'f1_loss': f1_loss})

    # model.evaluate()
    predictions = model.predict(pred_data_gen, verbose=1, steps=steps, max_queue_size=10, workers=1, use_multiprocessing=False)
    ckpt = model_file.split('_')[1].split('.')[0]
    data_utils.save_to_file(test_fname, predictions, decode=True, ckpt=ckpt)


if __name__ == '__main__':
    # data_partition(tv_ratio=0.2, ratio=1)
    # norm_params(train='../Data/tra_aug.csv', val='../Data/val_aug.csv', full='../Data/full_aug.csv')
    # train_pipline(train='../Data/tra.csv', val='../Data/val.csv', full='../Data/full.csv', ckpt=None)

    for model_num in range(20, 220, 20):
        model_evaluation(model_file='./model/model_{}.h5'.format(str(model_num)))

    # image = get_image('000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0-aug_13009', '../Data/train_s')
    # print(image.shape)
    # im = Image.open('../Data/train_s/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0-aug_13009_red.png')
    # print(np.asarray(im.con vert('RGB')).shape)

    # tf.keras.layers.GlobalMaxPool1D()
    # tf.keras.applications.resnet50.ResNet50()

