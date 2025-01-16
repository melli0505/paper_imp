import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from attention import MultiHeadAttention


class PositionwiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ffn=2048):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn

        self.dense1 = layers.Dense(self.d_ffn, activation="relu")
        self.dense2 = layers.Dense(self.d_model)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ffn, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        # Multi-Head Attention
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)

        # FeedForward Network
        self.feedforward = PositionwiseFeedForward(d_model, d_ffn)

        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

        # Layer Normalization
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, mask, training):
        attn_output, _ = self.multi_head_attention(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)

        ffn_output = self.feedforward(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ffn, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        # Masked Multi-head Attention
        self.masked_multi_head_attention = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads
        )

        # Multi-Head Attention
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads
        )

        # FeedForward Network
        self.feedforward = PositionwiseFeedForward(d_model=d_model, d_ffn=d_ffn)

        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

        # Layer Normalization
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm3 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, encoder_output, look_ahead_mask, padding_mask, training):
        # Masked Multi-Head Attention
        attn1, _ = self.masked_multi_head_attention(x, x, x, mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layer_norm1(x + attn1)

        attn2, _ = self.multi_head_attention(
            out1, encoder_output, encoder_output, mask=padding_mask
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layer_norm2(out1 + attn2)

        ffn_output = self.feedforward(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layer_norm3(out2 + ffn_output)

        return out3
