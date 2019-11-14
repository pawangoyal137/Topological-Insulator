import tensorflow as tf
import numpy as np
import sonnet as snt

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

def naive(pos, ids, d):
    """
    A naive RNN model to deal with dynamic length
    """

    W_embed = tf.get_variable(name='Embedding', shape=[d, 32], dtype=tf.float32)
    h_embed = tf.nn.embedding_lookup(W_embed, ids)

    h = tf.concat([h_embed, pos], axis=2)

    cell = tf.nn.rnn_cell.BasicLSTMCell(32, state_is_tuple=True, forget_bias=1)
    hidden_out, _ = tf.nn.dynamic_rnn(cell, h, dtype=tf.float32)
    out = hidden_out[:,-1,:]
    return tf.layers.dense(out, 4, activation=None)


def tinet(input_graph):
    # W_embed = tf.get_variable(name='Embedding', shape=[d, 32], dtype=tf.float32)
    # h_embed = tf.nn.embedding_lookup(W_embed, ids)

    # h = tf.concat([h_embed, pos], axis=2)
    
    node_block1 = blocks.NodeBlock(
        node_model_fn=lambda: snt.Linear(output_size=16)   
    )
    global_block1 = blocks.GlobalBlock(
        global_model_fn=lambda: snt.Linear(output_size=4)   
    )

    node_block2 = blocks.NodeBlock(
        node_model_fn=lambda: snt.Linear(output_size=32)   
    )
    global_block2 = blocks.GlobalBlock(
        global_model_fn=lambda: snt.Linear(output_size=8)   
    )

    node_block3 = blocks.NodeBlock(
        node_model_fn=lambda: snt.Linear(output_size=64)   
    )
    global_block3 = blocks.GlobalBlock(
        global_model_fn=lambda: snt.Linear(output_size=16)   
    )

    h1 = global_block1(node_block1(input_graph))
    h2 = global_block2(node_block2(h1))
    h3 = global_block3(node_block3(h2))

    out = h3.globals

    return tf.layers.dense(out, 4, activation=None)
