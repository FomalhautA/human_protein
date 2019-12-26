import tensorflow as tf


def inception_init(out1, r3, out3, r5, out5, outm):
    res = dict()

    res['out1'] = out1
    res['r3'] = r3
    res['out3'] = out3
    res['r5'] = r5
    res['out5'] = out5
    res['outm'] = outm

    return res


def arch_stone():
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

    return conv_params


def inception_v2(inputs, params, training=False):
    out1 = params['out1']
    r3 = params['r3']
    out3 = params['out3']
    r5 = params['r5']
    out5 = params['out5']
    outm = params['outm']

    z1 = tf.layers.conv2d(inputs=inputs, filters=out1, kernel_size=(1, 1), strides=(1, 1), padding='same',
                          activation=None)

    z_r3 = tf.layers.conv2d(inputs=inputs, filters=r3, kernel_size=(1, 1), strides=(1, 1), padding='same',
                            activation=None)

    z3 = tf.layers.conv2d(inputs=z_r3, filters=out3, kernel_size=(3, 3), strides=(1, 1), padding='same',
                          activation=None)

    z_r5 = tf.layers.conv2d(inputs=inputs, filters=r5, kernel_size=(1, 1), strides=(1, 1), padding='same',
                            activation=None)

    z5_1 = tf.layers.conv2d(inputs=z_r5, filters=out5, kernel_size=(3, 3), strides=(1, 1), padding='same',
                            activation=None)

    z5_2 = tf.layers.conv2d(inputs=z5_1, filters=out5, kernel_size=(3, 3), strides=(1, 1), padding='same',
                            activation=None)

    m = tf.layers.max_pooling2d(inputs=inputs, pool_size=(3, 3), strides=(1, 1), padding='same')

    zm = tf.layers.conv2d(inputs=m, filters=outm, kernel_size=(1, 1), strides=(1, 1), padding='same',
                          activation=None)

    z = tf.concat(values=[z1, z3, z5_2, zm], axis=3)

    bn = tf.layers.batch_normalization(z, axis=-1, training=training)

    return tf.nn.leaky_relu(bn)


def forward(X, params, mode, fc1=1000, output=28):
    # training = True if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL else False
    training = True

    # Layer 1
    z1 = tf.layers.conv2d(inputs=X, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                          activation=None)
    bn1 = tf.layers.batch_normalization(z1, axis=-1, training=training)
    a1 = tf.nn.leaky_relu(bn1)

    m1 = tf.layers.max_pooling2d(inputs=a1, pool_size=(3, 3), strides=(2, 2), padding="same")

    # Layer 2
    z_2r = tf.layers.conv2d(inputs=m1, filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                            activation=tf.nn.leaky_relu)
    # ToDo: Maybe need batch norm.

    z2 = tf.layers.conv2d(inputs=z_2r, filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same',
                          activation=None)
    bn2 = tf.layers.batch_normalization(z2, axis=-1, training=training)
    a2 = tf.nn.leaky_relu(bn2)

    # Max Pool
    m2 = tf.layers.max_pooling2d(inputs=a2, pool_size=(3, 3), strides=(2, 2), padding="same")

    # Layer 3a
    a_3a = inception_v2(m2, params["c_L3a"], training=training)
    # layer 3b
    a_3b = inception_v2(a_3a, params["c_L3b"], training=training)

    # Max Pool
    m3 = tf.layers.max_pooling2d(inputs=a_3b, pool_size=(3, 3), strides=(2, 2), padding='same')

    # Layer 4a
    a_4a = inception_v2(m3, params["c_L4a"], training=training)
    # Layer 4b
    a_4b = inception_v2(a_4a, params["c_L4b"], training=training)
    # Layer 4c
    a_4c = inception_v2(a_4b, params["c_L4c"], training=training)
    # Layer 4d
    a_4d = inception_v2(a_4c, params["c_L4d"], training=training)
    # Layer 4e
    a_4e = inception_v2(a_4d, params["c_L4e"], training=training)

    # Max Pool
    m4 = tf.layers.max_pooling2d(inputs=a_4e, pool_size=(3, 3), strides=(2, 2), padding='same')

    # Layer 5a
    a_5a = inception_v2(m4, params["c_L5a"], training=training)
    # Layer 5b
    a_5b = inception_v2(a_5a, params["c_L5b"], training=training)

    # Avg Pool
    ap = tf.layers.average_pooling2d(inputs=a_5b, pool_size=(8, 8), strides=(1, 1), padding='valid')

    # tf.keras.losses.CategoricalCrossentropy()
    # Full Connect
    z_fc = tf.layers.dense(tf.reshape(tf.squeeze(ap), [-1, 1024]), units=output, activation=None)
    bnfc = tf.layers.batch_normalization(z_fc, axis=-1, training=training)
    a_out = tf.nn.sigmoid(bnfc, name="logits")

    #     # Full Connect
    #     z_out = tf.contrib.layers.fully_connected(a_fc, num_outputs=output, activation_fn=None)
    # #     bnout = batch_norm(z_out, params["c_Lout_scale"], params["c_Lout_beta"], epsilon)
    #     a_out = tf.nn.sigmoid(z_out)

    return a_out
