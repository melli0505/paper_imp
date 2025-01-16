import tensorflow as tf


def create_padding_mask(seq):
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]  # shape = (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
    tf.ones((4,4)) -> [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    tf.linalg.band_part,
      num_lower = -1 : 대각선 아래 모든 요소를 남김
      num_upper = 0 : 대각선 위 모든 요소를 0으로 처리리 -> [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]
    1 - tf.linalg.band_part(~) : 에서 빼줘서 값 반전 -> [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
    1로 마스킹됨 -> 나중에 1e-6 이랑 곱해져서 참조 못하게 될 예정
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), num_lower=-1, num_upper=0)
    return mask


def create_masks(input_seq, target_seq):
    encoder_padding_mask = create_padding_mask(input_seq)
    decoder_padding_mask = create_padding_mask(input_seq)
    look_ahead_mask = create_look_ahead_mask(tf.shape(target_seq)[1])
    target_padding_mask = create_padding_mask(target_seq)

    combined_mask = tf.maximum(look_ahead_mask, target_padding_mask)

    return encoder_padding_mask, decoder_padding_mask, combined_mask
