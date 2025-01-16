import tensorflow as tf
import numpy as np


class CustomDataLoader(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, shuffle=False):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # 전체 dataset 길이 / batch size 가 loader의 길이가 됨
        return int(tf.math.ceil(len(self.x) / self.batch_size))  # ceil: 올림

    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = [self.x[i] for i in indices]
        batch_y = [self.y[i] for i in indices]

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indices)
