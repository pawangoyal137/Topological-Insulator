import tensorflow as tf
import numpy as np
import sonnet as snt

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

def naive(pos, ids, seq_len,num_layers):
    """
    A naive RNN model to deal with dynamic length
    """

    W_embed = tf.get_variable(name='Embedding', shape=[800, 32], dtype=tf.float32)
    h_embed = tf.nn.embedding_lookup(W_embed, ids)

    h = tf.concat([h_embed, pos], axis=2)
    cell=[tf.nn.rnn_cell.BasicLSTMCell(32, state_is_tuple=True, forget_bias=1) for i in range(num_layers)]
    cell = tf.nn.rnn_cell.MultiRNNCell(cell, state_is_tuple=True)
    hidden_out, _ = tf.nn.dynamic_rnn(cell, h, dtype=tf.float32, sequence_length=seq_len)
    print('bbbbbbb',hidden_out)
    out = tf.gather(hidden_out, seq_len-1, axis=1)
    out = out[:,-1,:]
    return tf.layers.dense(out, 4, activation=None)


def tinet(input_graph):
    embedding = blocks.NodeBlock(
        node_model_fn=lambda: tf.keras.layers.Embedding(800, 32),
        use_received_edges=False,
        use_sent_edges=False,
        use_nodes=True,
        use_globals=False)
    graph_network_layer1 = modules.GraphNetwork(
        edge_model_fn = lambda: tf.layers.Dense(16, activation=tf.tanh),
        node_model_fn = lambda: tf.layers.Dense(16, activation=tf.tanh),
        global_model_fn = lambda: tf.layers.Dense(8, activation=tf.tanh))
    graph_network_layer2 = modules.GraphNetwork(
        edge_model_fn = lambda: tf.layers.Dense(16, activation=tf.tanh),
        node_model_fn = lambda: tf.layers.Dense(16, activation=tf.tanh),
        global_model_fn = lambda: tf.layers.Dense(8, activation=tf.tanh))
    graph_network_layer3 = modules.GraphNetwork(
        edge_model_fn = lambda: tf.layers.Dense(16, activation=tf.tanh),
        node_model_fn = lambda: tf.layers.Dense(16, activation=tf.tanh),
        global_model_fn = lambda: tf.layers.Dense(8, activation=tf.tanh))
    graph_network_layer4 = modules.GraphNetwork(
        edge_model_fn = lambda: tf.layers.Dense(16, activation=tf.tanh),
        node_model_fn = lambda: tf.layers.Dense(16, activation=tf.tanh),
        global_model_fn = lambda: tf.layers.Dense(8, activation=tf.tanh))
    graph_network_layer5 = modules.GraphNetwork(
        edge_model_fn = lambda: tf.layers.Dense(16, activation=tf.tanh),
        node_model_fn = lambda: tf.layers.Dense(16, activation=tf.tanh),
        global_model_fn = lambda: tf.layers.Dense(8, activation=tf.tanh))
    graph_network_layer6 = modules.GraphNetwork(
        edge_model_fn = lambda: tf.layers.Dense(16, activation=tf.tanh),
        node_model_fn = lambda: tf.layers.Dense(16, activation=tf.tanh),
        global_model_fn = lambda: tf.layers.Dense(8, activation=tf.tanh))

    h0 = embedding(input_graph)
    h1 = graph_network_layer1(h0)
    h2 = graph_network_layer2(h1)
    h3 = graph_network_layer3(h2)
    h4 = graph_network_layer4(h3)
    h5 = graph_network_layer5(h4)
    h6 = graph_network_layer6(h5)

    out = h6.globals

    return tf.layers.dense(out, 4, activation=None)

# pos = tf.placeholder(tf.float32, [None, None, 3])
# ids = tf.placeholder(tf.int32, [None, None])
# lattice = tf.placeholder(tf.float32, [None, 3, 3])
# y = tf.placeholder(tf.int64, [None])
# seq_len = tf.placeholder(tf.int32, [None])

if __name__ == '__main__':
    naive(pos, ids, seq_len)