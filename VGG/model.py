import tensorflow as tf
from tensorflow.keras import layers


class VGG16(tf.keras.Model):
    def __init__(self, input_shape=(224, 224, 3), num_classes=100):
        super(VGG16, self).__init__()

        self.features = tf.keras.Sequential(
            [
                layers.Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="same",
                    name="conv1_1",
                    input_shape=input_shape,
                ),
                layers.Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="same",
                    name="conv1_2",
                ),
                layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool1"),
                layers.Conv2D(
                    filters=128,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="same",
                    name="conv2_1",
                ),
                layers.Conv2D(
                    filters=128,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="same",
                    name="conv2_2",
                ),
                layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool2"),
                layers.Conv2D(
                    filters=256,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="same",
                    name="conv3_1",
                ),
                layers.Conv2D(
                    filters=256,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="same",
                    name="conv3_2",
                ),
                layers.Conv2D(
                    filters=256,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="same",
                    name="conv3_3",
                ),
                layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool3"),
                layers.Conv2D(
                    filters=512,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="same",
                    name="conv4_1",
                ),
                layers.Conv2D(
                    filters=512,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="same",
                    name="conv4_2",
                ),
                layers.Conv2D(
                    filters=512,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="same",
                    name="conv4_3",
                ),
                layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool4"),
                layers.Conv2D(
                    filters=512,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="same",
                    name="conv5_1",
                ),
                layers.Conv2D(
                    filters=512,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="same",
                    name="conv5_2",
                ),
                layers.Conv2D(
                    filters=512,
                    kernel_size=(3, 3),
                    strides=1,
                    padding="same",
                    name="conv5_3",
                ),
                layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool5"),
                layers.Flatten(name="flatten"),
                layers.Dense(4096, name="fc1", activation="relu"),
                layers.Dropout(0.5, name="dropout1"),
                layers.Dense(4096, name="fc2", activation="relu"),
                layers.Dropout(0.5, name="dropout2"),
                layers.Dense(num_classes, name="fc3", activation="softmax"),
            ]
        )

    def get_layer_info(self):
        return self.features.summary()

    def call(self, x):
        output = self.features(x)
        return output
