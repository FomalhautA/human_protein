import tensorflow as tf
import tensorflow_addons as tfa

from loss import focal_loss, f1_loss


def inception_init(out1, r3, out3, r5, out5, outm):
    res = dict()

    res['out1'] = out1
    res['r3'] = r3
    res['out3'] = out3
    res['r5'] = r5
    res['out5'] = out5
    res['outm'] = outm

    return res


def inception_arch_stone():
    conv_params = dict()

    conv_params['c_L3a'] = inception_init(out1=64, r3=96, out3=128, r5=16, out5=32, outm=32)
    conv_params['c_L3b'] = inception_init(out1=128, r3=128, out3=192, r5=32, out5=96, outm=64)

    conv_params["c_L4a"] = inception_init(out1=192, r3=96, out3=208, r5=16, out5=48, outm=64)
    conv_params["c_L4b"] = inception_init(out1=160, r3=112, out3=224, r5=24, out5=64, outm=64)
    conv_params["c_L4c"] = inception_init(out1=128, r3=128, out3=256, r5=24, out5=64, outm=64)
    conv_params["c_L4d"] = inception_init(out1=112, r3=144, out3=288, r5=32, out5=64, outm=64)
    conv_params["c_L4e"] = inception_init(out1=256, r3=160, out3=320, r5=32, out5=128, outm=128)

    conv_params["c_L5a"] = inception_init(out1=256, r3=160, out3=320, r5=32, out5=128, outm=128)
    conv_params["c_L5b"] = inception_init(out1=384, r3=192, out3=384, r5=48, out5=128, outm=128)

    conv_params['c_L6a'] = inception_init(out1=384, r3=192, out3=384, r5=48, out5=128, outm=128)
    conv_params['c_L6b'] = inception_init(out1=512, r3=256, out3=512, r5=64, out5=128, outm=128)

    return conv_params


def resnet_arch_stone():
    conv_params = dict()

    conv_params['c_L3a'] = inception_init(out1=64, r3=32, out3=64, r5=16, out5=64, outm=64)
    # conv_params['c_L3b'] = inception_init(out1=128, r3=64, out3=128, r5=32, out5=128, outm=128)
    conv_params['c_L3b'] = inception_init(out1=64, r3=64, out3=64, r5=32, out5=64, outm=64)

    # conv_params["c_L4a"] = inception_init(out1=192, r3=64, out3=208, r5=32, out5=48, outm=64)
    # conv_params["c_L4b"] = inception_init(out1=160, r3=96, out3=224, r5=40, out5=64, outm=64)
    # conv_params["c_L4c"] = inception_init(out1=128, r3=128, out3=256, r5=48, out5=64, outm=64)
    # conv_params["c_L4d"] = inception_init(out1=112, r3=144, out3=272, r5=56, out5=64, outm=64)
    # conv_params["c_L4e"] = inception_init(out1=256, r3=160, out3=256, r5=64, out5=128, outm=128)

    conv_params["c_L4a"] = inception_init(out1=64, r3=64, out3=64, r5=32, out5=64, outm=64)
    conv_params["c_L4b"] = inception_init(out1=64, r3=96, out3=64, r5=40, out5=64, outm=64)
    conv_params["c_L4c"] = inception_init(out1=64, r3=128, out3=64, r5=48, out5=64, outm=64)
    conv_params["c_L4d"] = inception_init(out1=64, r3=144, out3=64, r5=56, out5=64, outm=64)
    conv_params["c_L4e"] = inception_init(out1=64, r3=160, out3=64, r5=64, out5=64, outm=64)

    # conv_params["c_L5a"] = inception_init(out1=256, r3=176, out3=352, r5=32, out5=128, outm=128)
    # conv_params["c_L5b"] = inception_init(out1=384, r3=192, out3=384, r5=48, out5=128, outm=128)

    conv_params["c_L5a"] = inception_init(out1=64, r3=176, out3=64, r5=32, out5=64, outm=64)
    conv_params["c_L5b"] = inception_init(out1=64, r3=192, out3=64, r5=48, out5=64, outm=64)

    # conv_params['c_L6a'] = inception_init(out1=384, r3=224, out3=384, r5=48, out5=128, outm=128)
    # conv_params['c_L6b'] = inception_init(out1=512, r3=256, out3=512, r5=64, out5=128, outm=128)

    conv_params['c_L6a'] = inception_init(out1=64, r3=224, out3=64, r5=48, out5=64, outm=64)
    conv_params['c_L6b'] = inception_init(out1=64, r3=256, out3=64, r5=64, out5=64, outm=64)

    return conv_params


def inception_v2(inputs, params):
    """

    :param inputs:
    :param params:
    :return:
    """
    z1 = tf.keras.layers.Conv2D(filters=params['out1'], kernel_size=(1, 1), strides=(1, 1), padding='same',
                                activation=None)(inputs)

    z_r3 = tf.keras.layers.Conv2D(filters=params['r3'], kernel_size=(1, 1), strides=(1, 1), padding='same',
                                  activation=None)(inputs)
    z3 = tf.keras.layers.Conv2D(filters=params['out3'], kernel_size=(3, 3), strides=(1, 1), padding='same',
                                activation=None)(z_r3)

    z_r5 = tf.keras.layers.Conv2D(filters=params['r5'], kernel_size=(1, 1), strides=(1, 1), padding='same',
                                  activation=None)(inputs)
    z5_1 = tf.keras.layers.Conv2D(filters=params['out5'], kernel_size=(3, 3), strides=(1, 1), padding='same',
                                  activation=None)(z_r5)
    z5 = tf.keras.layers.Conv2D(filters=params['out5'], kernel_size=(3, 3), strides=(1, 1), padding='same',
                                activation=None)(z5_1)

    m = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    zm = tf.keras.layers.Conv2D(filters=params['outm'], kernel_size=(1, 1), strides=(1, 1), padding='same',
                                activation=None)(m)

    z = tf.keras.layers.Concatenate(axis=-1)([z1, z3, z5, zm])

    bn = tf.keras.layers.BatchNormalization(axis=-1)(z)

    return tf.keras.layers.LeakyReLU()(bn)


def resnet_block(inputs, params):
    z1 = tf.keras.layers.Conv2D(filters=params['out1'], kernel_size=(1, 1), strides=(1, 1), padding='same',
                                activation=None)(inputs)

    z_r3 = tf.keras.layers.Conv2D(filters=params['r3'], kernel_size=(1, 1), strides=(1, 1), padding='same',
                                  activation=None)(inputs)
    z3 = tf.keras.layers.Conv2D(filters=params['out3'], kernel_size=(3, 3), strides=(1, 1), padding='same',
                                activation=None)(z_r3)

    z_r5 = tf.keras.layers.Conv2D(filters=params['r5'], kernel_size=(1, 1), strides=(1, 1), padding='same',
                                  activation=None)(inputs)
    z5_1 = tf.keras.layers.Conv2D(filters=params['out5'], kernel_size=(3, 3), strides=(1, 1), padding='same',
                                  activation=None)(z_r5)
    z5 = tf.keras.layers.Conv2D(filters=params['out5'], kernel_size=(3, 3), strides=(1, 1), padding='same',
                                activation=None)(z5_1)

    m = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    zm = tf.keras.layers.Conv2D(filters=params['outm'], kernel_size=(1, 1), strides=(1, 1), padding='same',
                                activation=None)(m)

    z = tf.keras.layers.Concatenate(axis=-1)([z1, z3, z5, zm])

    bn = tf.keras.layers.BatchNormalization(axis=-1)(z)

    incep = tf.keras.layers.LeakyReLU()(bn)

    re = tf.add(incep, inputs)
    return re


def create_model_inception(params, output=28, lr=0.02, batch_size=8, lr_decay=0.999, decay_steps=6000):
    """

    :param params:
    :param output:
    :param lr:
    :batch_size:
    :return:
    """
    inputs = tf.keras.layers.Input(shape=(256, 256, 4), batch_size=batch_size, dtype=tf.dtypes.float32)

    # Layer 1
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    # Layer 2
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation=None)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    # ToDo: Maybe need batch norm.

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Max Pool
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    # Layer 3a
    x = inception_v2(x, params["c_L3a"])
    # layer 3b
    x = inception_v2(x, params["c_L3b"])

    # Max Pool
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Layer 4a
    x = inception_v2(x, params["c_L4a"])
    # Layer 4b
    x = inception_v2(x, params["c_L4b"])
    # Layer 4c
    x = inception_v2(x, params["c_L4c"])
    # Layer 4d
    x = inception_v2(x, params["c_L4d"])
    # Layer 4e
    x = inception_v2(x, params["c_L4e"])

    # Max Pool
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Layer 5a
    x = inception_v2(x, params["c_L5a"])
    # Layer 5b
    x = inception_v2(x, params["c_L5b"])

    # # Max Pool
    # m5 = tf.layers.max_pooling2d(inputs=a_5b, pool_size=(3, 3), strides=(2, 2), padding='same')
    #
    # # Layer 6a
    # a_6a = inception_v2(m5, params['c_L6a'], training=training)
    # # Layer 6b
    # a_6b = inception_v2(a_6a, params['c_L6b'], training=training)

    # Avg Pool
    x = tf.keras.layers.AvgPool2D(pool_size=(8, 8), strides=(1, 1), padding='valid')(x)

    # tf.keras.losses.CategoricalCrossentropy()
    # Full Connect
    x = tf.keras.layers.Dense(units=1000, activation=None)(tf.reshape(tf.squeeze(x), [-1, 1024]))
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

    # Full Connect
    x = tf.keras.layers.Dense(units=output, activation=None)(tf.reshape(tf.squeeze(x), [-1, 1000]))
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(x)

    model = tf.keras.Model(inputs, x)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=decay_steps,
                                                                 decay_rate=lr_decay, staircase=True)

    # lr_schedule = tf.keras.optimizers.schedules.LearningRateSchedule()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=f1_loss,
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tfa.metrics.F1Score(num_classes=output, average='macro')],
                  loss_weights=None, sample_weight_mode=None, weighted_metrics=None)

    return model


def create_model_resnet(params, output=28, lr=0.02, batch_size=8, lr_decay=0.999, decay_steps=6000):
    """

    :param params:
    :param output:
    :param lr:
    :batch_size:
    :return:
    """
    inputs = tf.keras.layers.Input(shape=(256, 256, 4), batch_size=batch_size, dtype=tf.dtypes.float32)

    # Layer 1
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    # Layer 2
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation=None)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    # ToDo: Maybe need batch norm.

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Max Pool
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    # Layer 3a
    x = resnet_block(x, params["c_L3a"])
    # layer 3b
    x = resnet_block(x, params["c_L3b"])

    # Max Pool
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Layer 4a
    x = resnet_block(x, params["c_L4a"])
    # Layer 4b
    x = resnet_block(x, params["c_L4b"])
    # Layer 4c
    x = resnet_block(x, params["c_L4c"])
    # Layer 4d
    x = resnet_block(x, params["c_L4d"])
    # Layer 4e
    x = resnet_block(x, params["c_L4e"])

    # Max Pool
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Layer 5a
    x = resnet_block(x, params["c_L5a"])
    # Layer 5b
    x = resnet_block(x, params["c_L5b"])

    # # Max Pool
    # m5 = tf.layers.max_pooling2d(inputs=a_5b, pool_size=(3, 3), strides=(2, 2), padding='same')
    #
    # # Layer 6a
    # a_6a = inception_v2(m5, params['c_L6a'], training=training)
    # # Layer 6b
    # a_6b = inception_v2(a_6a, params['c_L6b'], training=training)

    # Avg Pool
    x = tf.keras.layers.AvgPool2D(pool_size=(8, 8), strides=(1, 1), padding='valid')(x)

    # tf.keras.losses.CategoricalCrossentropy()
    # Full Connect
    x = tf.keras.layers.Dense(units=1000, activation=None)(tf.reshape(tf.squeeze(x), [-1, 256]))
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

    # Full Connect
    x = tf.keras.layers.Dense(units=output, activation=None)(tf.reshape(tf.squeeze(x), [-1, 1000]))
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(x)

    model = tf.keras.Model(inputs, x)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=decay_steps,
                                                                 decay_rate=lr_decay, staircase=True)

    # lr_schedule = tf.keras.optimizers.schedules.LearningRateSchedule()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=f1_loss,
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tfa.metrics.F1Score(num_classes=output, average='macro')],
                  loss_weights=None, sample_weight_mode=None, weighted_metrics=None)

    return model
