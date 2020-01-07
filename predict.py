import os
import model
import graph_nets as gn
import numpy as np
import pickle
import tensorflow as tf
from pymatgen.io.cif import CifParser


from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

from train import build_dict


# tf.app.flags.DEFINE_string('mode', 'graph', 'The name of the RNN model: graph, rnn')

# FLAGS = tf.app.flags.FLAGS


def softmaxAndName(p, filename):
    e_x = np.exp(p - np.max(p))
    pp = np.max(e_x / e_x.sum())

    parser = CifParser(filename, occupancy_tolerance=100)
    name = parser.get_structures()[0].composition.reduced_formula

    return pp, name

def main(_):


    with open('data/predict.pkl', 'rb') as f:
        predict_graphs = pickle.load(f)
    f.close()
    
    modified_graphs, _, _ = build_dict(predict_graphs[:1], 'graph')
    input_graph = utils_tf.placeholders_from_data_dicts(modified_graphs[0:1])
    y = tf.placeholder(tf.int64, [None])
    lattice = tf.placeholder(tf.float32, [None, 3, 3])
    h_hat = model.tinet(input_graph)

    # merge lattice information w/ atoms
    h_lattice = tf.reshape(lattice, [-1, 9])
    h = tf.concat([h_hat, h_lattice], axis=1)
    y_hat = tf.layers.dense(h, 4, activation=None)


    # load model
    saver = tf.train.Saver()

    graph_ti = []
    graph_tci = []

    # sess predict
    with tf.Session() as sess:
        saver.restore(sess, "model/" + "graph" + "/model.ckpt")

        for i in range(len(predict_graphs)):
            batch_graphnets, batch_labels, batch_lattice = build_dict([predict_graphs[i]], 'graph')
            if not batch_labels:
                continue
            test_batch_graph_data = utils_np.data_dicts_to_graphs_tuple(batch_graphnets)
            feed_dict = {input_graph: test_batch_graph_data, y: batch_labels, 
                    lattice: batch_lattice}
            y_hat_value = sess.run(y_hat, feed_dict=feed_dict)[0]
            y_value = np.argmax(y_hat_value)

            if y_value == 3 and predict_graphs[i]['y'] == 0:
                graph_ti.append(softmaxAndName(y_hat_value, predict_graphs[i]['name']))
            if y_value == 2 and predict_graphs[i]['y'] == 0:
                graph_tci.append(softmaxAndName(y_hat_value, predict_graphs[i]['name']))


    graph_ti.sort(reverse=True)            
    graph_tci.sort(reverse=True)            

    tf.compat.v1.reset_default_graph()

    with open('data/predict.pkl', 'rb') as f:
        predict_graphs = pickle.load(f)
    f.close()
    
    pos = tf.placeholder(tf.float32, [None, None, 3])
    ids = tf.placeholder(tf.int32, [None, None])
    lattice = tf.placeholder(tf.float32, [None, 3, 3])
    y = tf.placeholder(tf.int64, [None])
    seq_len = tf.placeholder(tf.int32, [None])
    h_hat = model.naive(pos, ids, seq_len, 2)

    # merge lattice information w/ atoms
    h_lattice = tf.reshape(lattice, [-1, 9])
    h = tf.concat([h_hat, h_lattice], axis=1)
    y_hat = tf.layers.dense(h, 4, activation=None)


    # load model
    saver = tf.train.Saver()

    rnn_ti = []
    rnn_tci = []

    # sess predict
    with tf.Session() as sess:
        saver.restore(sess, "model/" + "rnn" + "/model.ckpt")

        for i in range(len(predict_graphs)):
            batch_pos, batch_ids, batch_lattice, batch_y, batch_seq_len = build_dict([predict_graphs[i]], 'rnn')
            feed_dict = {pos: batch_pos, ids: batch_ids, 
                                lattice: batch_lattice, y:batch_y, seq_len:batch_seq_len}
            y_hat_value = sess.run(y_hat, feed_dict=feed_dict)[0]
            y_value = np.argmax(y_hat_value)

            if y_value == 3 and predict_graphs[i]['y'] == 0:
                rnn_ti.append(softmaxAndName(y_hat_value, predict_graphs[i]['name']))
            if y_value == 2 and predict_graphs[i]['y'] == 0:
                rnn_tci.append(softmaxAndName(y_hat_value, predict_graphs[i]['name']))


    rnn_ti.sort(reverse=True)            
    rnn_tci.sort(reverse=True)            

    tf.compat.v1.reset_default_graph()


    overall_ti = {}
    for i in rnn_ti:
        p, v = i
        if v in overall_ti:
            overall_ti[v] += p
        else:
            overall_ti[v] = p
    for i in graph_ti:
        p, v = i
        if v in overall_ti:
            overall_ti[v] += p
        else:
            overall_ti[v] = p
    ti = []
    for k in overall_ti:
        ti.append((overall_ti[k], k))
    ti.sort(reverse=True)


    overall_tci = {}
    for i in rnn_tci:
        p, v = i
        if v in overall_tci:
            overall_tci[v] += p
        else:
            overall_tci[v] = p
    for i in graph_tci:
        p, v = i
        if v in overall_tci:
            overall_tci[v] += p
        else:
            overall_tci[v] = p
    tci = []
    for k in overall_tci:
        tci.append((overall_tci[k], k))
    tci.sort(reverse=True)

    for k in ti:
        print(k)
    print("############")
    for k in tci:
        print(k)


if __name__ == '__main__':
    tf.app.run()