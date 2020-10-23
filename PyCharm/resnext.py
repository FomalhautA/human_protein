import tensorflow as tf
import tensorflow_addons as tfa

from loss import f1_loss


def resnext_block(inputs, output_channels, cardinality=32):
    """

    :param inputs:
    :param input_channels:
    :param output_channels:
    :param cardinality:
    :return:
    """
    covs = []
    for i in range(cardinality):
        temp = tf.keras.layers.Conv2D(filters=output_channels//2//cardinality, kernel_size=(1, 1), strides=(1, 1),
                                      padding='same', activation=None)(inputs)
        temp = tf.keras.layers.Conv2D(filters=output_channels//2//cardinality, kernel_size=(3, 3), strides=(1, 1),
                                      padding='same', activation=None)(temp)
        temp = tf.keras.layers.Conv2D(filters=output_channels, kernel_size=(1, 1), strides=(1, 1),
                                      padding='same', activation=None)(temp)
        covs.append(temp)

    z1 = tf.add_n(covs)
    bn = tf.keras.layers.BatchNormalization(axis=-1)(z1)
    a1 = tf.keras.layers.LeakyReLU()(bn)
    re = tf.add(inputs, a1)

    return re


def create_model_resnext(output=28, lr=0.02, batch_size=8, lr_decay=0.999, decay_steps=6000):
    """

    :param output:
    :param lr:
    :param batch_size:
    :param lr_decay:
    :param decay_steps:
    :return:
    """
    inputs = tf.keras.layers.Input(shape=(256, 256, 4), batch_size=batch_size, dtype=tf.dtypes.float32)

    # Layer 1
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    # Layer Conv2-1
    x = resnext_block(x, output_channels=256, cardinality=32)
    # Layer Conv2-2
    x = resnext_block(x, output_channels=256, cardinality=32)
    # Layer Conv2-3
    x = resnext_block(x, output_channels=256, cardinality=32)

    # Layer 2 to 3
    x = tf.keras.layers.MaxPool2D(kernel_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Layer Conv3-1
    x = resnext_block(x, output_channels=512, cardinality=32)
    # Layer Conv3-2
    x = resnext_block(x, output_channels=512, cardinality=32)
    # Layer Conv3-3
    x = resnext_block(x, output_channels=512, cardinality=32)
    # Layer Conv3-4
    x = resnext_block(x, output_channels=512, cardinality=32)

    # Layer 3 to 4
    x = tf.keras.layers.MaxPool2D(kernel_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Layer Conv4-1
    x = resnext_block(x, output_channels=1024, cardinality=32)
    # Layer Conv4-2
    x = resnext_block(x, output_channels=1024, cardinality=32)
    # Layer Conv4-3
    x = resnext_block(x, output_channels=1024, cardinality=32)
    # Layer Conv4-4
    x = resnext_block(x, output_channels=1024, cardinality=32)
    # Layer Conv4-5
    x = resnext_block(x, output_channels=1024, cardinality=32)
    # Layer Conv4-6
    x = resnext_block(x, output_channels=1024, cardinality=32)

    # # Layer 4 to 5
    x = tf.keras.layers.MaxPool2D(kernel_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Layer Conv5-1
    x = resnext_block(x, output_channels=2048, cardinality=32)
    # Layer Conv5-2
    x = resnext_block(x, output_channels=2048, cardinality=32)
    # Layer Conv5-3
    x = resnext_block(x, output_channels=2048, cardinality=32)

    # Avg Pool
    x = tf.keras.layers.AvgPool2D(pool_size=(8, 8), strides=(1, 1), padding='valid')(x)

    # tf.keras.losses.CategoricalCrossentropy()
    # Full Connect
    x = tf.keras.layers.Dense(units=1000, activation=None)(tf.squeeze(x))
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

    # Full Connect
    x = tf.keras.layers.Dense(units=output, activation=None)(tf.squeeze(x))
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
