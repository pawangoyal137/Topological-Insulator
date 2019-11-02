import graph_nets as gn
import sonnet as snt
import tensorflow as tf
import numpy as np


def naive(pos, ids, d):
    print(pos)
    print(ids)

    W_embed = tf.get_variable(name='Embedding', shape=[d, 32], dtype=tf.float32)
    h_embed = tf.nn.embedding_lookup(W_embed, ids)

    h = tf.concat([h_embed, pos], axis=2)
    print(h)

    cell = tf.nn.rnn_cell.BasicLSTMCell(32, state_is_tuple=True, forget_bias=1)
    hidden_out, _ = tf.nn.dynamic_rnn(cell, h, dtype=tf.float32)
    out = hidden_out[:,-1,:]
    print(out)
    return tf.layers.dense(out, 4, activation=None)


def tinet(x, num_layer=4, EDGE_SIZE=3, NODE_SIZE=3, GLOBAL_SIZE=3):
    pass


    
