import tensorflow as tf
import numpy as np
import sonnet as snt

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

def naive(pos, ids, seq_len):
    """
    A naive RNN model to deal with dynamic length
    """

    W_embed = tf.get_variable(name='Embedding', shape=[800, 32], dtype=tf.float32)
    h_embed = tf.nn.embedding_lookup(W_embed, ids)

    h = tf.concat([h_embed, pos], axis=2)

    cell = tf.nn.rnn_cell.BasicLSTMCell(32, state_is_tuple=True, forget_bias=1)
    hidden_out, _ = tf.nn.dynamic_rnn(cell, h, dtype=tf.float32, sequence_length=seq_len)
    out = tf.gather(hidden_out, seq_len-1, axis=1)
    out = out[:,-1,:]
    return tf.layers.dense(out, 4, activation=None)


def tinet(input_graph):
    # W_embed = tf.get_variable(name='Embedding', shape=[d, 32], dtype=tf.float32)
    # h_embed = tf.nn.embedding_lookup(W_embed, ids)

    # h = tf.concat([h_embed, pos], axis=2)

    graph_network_layer1 = modules.GraphNetwork(
        edge_model_fn=lambda: snt.Linear(output_size=32),
        node_model_fn=lambda: snt.Linear(output_size=16),
        global_model_fn=lambda: snt.Linear(output_size=4))

    graph_network_layer2 = modules.GraphNetwork(
        edge_model_fn=lambda: snt.Linear(output_size=32),
        node_model_fn=lambda: snt.Linear(output_size=16),
        global_model_fn=lambda: snt.Linear(output_size=4))

    graph_network_layer3 = modules.GraphNetwork(
        edge_model_fn=lambda: snt.Linear(output_size=32),
        node_model_fn=lambda: snt.Linear(output_size=16),
        global_model_fn=lambda: snt.Linear(output_size=4))
    
    h1 = graph_network_layer1(input_graph)
    h2 = graph_network_layer2(h1)
    h3 = graph_network_layer3(h2)

    out = h3.globals

    return tf.layers.dense(out, 4, activation=None)
