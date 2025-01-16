import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from layers import EncoderLayer, DecoderLayer


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self, num_layers, d_model, num_heads, d_ffn, vocab_size, dropout_rate=0.1
    ):
        super(Encoder, self).__init__()

        # 입력 임베딩
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=d_model
        )  # 단어집의 크기가 10000일 경우

        # Positional Encoding
        self.pos_encoding = self.add_weight(
            "pos_encoding", shape=[1, 10000, d_model], trainable=False
        )

        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, d_ffn, dropout_rate)
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, mask, training):
        # embedding + positional encoding
        x = self.embedding(x) + self.pos_encoding[:, : tf.shape(x)[1], :]
        x = self.dropout(x, training=training)
        for layer in self.encoder_layers:
            x = layer(x, mask, training)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self, num_layers, d_model, num_heads, d_ffn, vocab_size, dropout_rate=0.1
    ):
        super(Decoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=d_model
        )
        self.pos_encoding = self.add_weight(
            "pos_encoding", shape=[1, 10000, d_model], trainable=False
        )

        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, d_ffn, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, encoder_output, look_ahead_mask, mask, training):
        # embedding + positional encoding
        x = self.embedding(x) + self.pos_encoding[:, : tf.shape(x)[1], :]
        x = self.dropout(x, training=training)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, look_ahead_mask, mask, training)
        return x


class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_enc_layers,
        num_dec_layers,
        d_model,
        num_heads,
        d_ffn,
        input_vocab_size,
        target_vocab_size,
        max_pos_enc,
        dropout_rate=0.1,
    ):
        super(Transformer, self).__init__()

        # Encoder
        self.encoder = Encoder(
            num_enc_layers, d_model, num_heads, d_ffn, input_vocab_size, dropout_rate
        )

        # Decoder
        self.decoder = Decoder(
            num_dec_layers, d_model, num_heads, d_ffn, target_vocab_size, dropout_rate
        )

        # output
        self.dense = layers.Dense(target_vocab_size)

    def call(
        self,
        enc_input,
        dec_input,
        encoder_padding_mask,
        look_ahead_mask,
        decoder_padding_mask,
        training,
    ):
        # Encoder
        encoder_output = self.encoder(enc_input, encoder_padding_mask, training)
        # def call(self, x, encoder_output, look_ahead_mask, mask, training):
        # Decoder
        decoder_output = self.decoder(
            dec_input, encoder_output, look_ahead_mask, decoder_padding_mask, training
        )

        # output
        output = self.dense(decoder_output)

        return output
