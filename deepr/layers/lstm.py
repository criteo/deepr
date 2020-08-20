# pylint: disable=no-value-for-parameter,unexpected-keyword-arg
"""LSTM layers."""

import tensorflow as tf

from deepr.layers import base


@base.layer(n_in=2, n_out=3)
def LSTM(tensors, num_units: int, bidirectional: bool = False, **kwargs):
    """LSTM layer."""
    words, nwords = tensors
    t = tf.transpose(words, perm=[1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=num_units, **kwargs)
    outputs_fw, (hidden_fw, output_fw) = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)

    if bidirectional:
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=num_units, **kwargs)
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        outputs_bw, (hidden_bw, output_bw) = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
        outputs = tf.concat([outputs_fw, outputs_bw], axis=-1)
        hidden = tf.concat([hidden_fw, hidden_bw], axis=-1)
        output = tf.concat([output_fw, output_bw], axis=-1)
    else:
        outputs = outputs_fw
        hidden = hidden_fw
        output = output_fw

    outputs = tf.transpose(outputs, perm=[1, 0, 2])
    return (outputs, hidden, output)
