import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def scaled_dot_product_attention(query, key, value, mask=None):
    # QK^T
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scaling
    d_k = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)

    # mask 적용
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # softmax
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # multiply with V
    output = tf.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        # self.depth = (
        #     d_model // self.num_heads
        # )  # head 별로 병렬 처리를 위해 사용될 길이를 의미함, depth

        self.depth = 16

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.depth)
        )  # num_head * depth = d_model
        # (batch_size, num_head, seq_len, depth)
        # num_head가 앞에 오도록 해서 각 head가 독립적으로 attention 수행하기 쉽도록
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]

        # linear projection
        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot product attention
        scaled_attention, attention_weights = scaled_dot_product_attention(
            query, key, value, mask
        )

        # concatenate head
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # 나눴던 헤드 다시 병합
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # final linear layer
        output = self.dense(concat_attention)

        return output, attention_weights
