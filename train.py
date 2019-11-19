import os
import model
import graph_nets as gn
import numpy as np
import pickle
import tensorflow as tf

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

import tensorflow.python.util as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

tf.app.flags.DEFINE_string(
    'model', 'graph', 'The name of the RNN model: graph, rnn')

FLAGS = tf.app.flags.FLAGS


def build_dict(graphs, model):
    if model == "rnn":
        # max_length
        max_len = 0
        for i in graphs:
            max_len = max(max_len, len(i["atoms"]))

        y = []
        lattice = []
        for i in graphs:
            y.append(i['y'])
            lattice.append(i['lattice'])

        pos = []
        ids = []
        seq_len = []
        for i in graphs:
            pos_empty = np.zeros((max_len-len(i['atoms']), 3))
            this_pos = np.array(i['coords'])
            this_pos = np.concatenate([this_pos, pos_empty], axis=0)
            pos.append(this_pos)

            ats_empty = np.zeros((max_len-len(i['atoms'])))
            this_ats = np.array(i['atoms'])
            this_ats = np.concatenate([this_ats, ats_empty], axis=0)
            ids.append(this_ats)

            seq_len.append(len(i['atoms']))
        return pos, ids, lattice, y, seq_len

    elif model == "graphs":
        graph_dicts = []
        labels = []
        for i in range(len(pos)):
            n = pos[i]
            if len(n) <= 1:
                print("Empty")
                continue

            emb = np.zeros((len(n), d))
            for k in range(len(n)):
                emb[k][ids[k]] = 1.
            n = np.concatenate([n, emb], axis=1).astype(np.float32)

            graph = {
                    "globals": [0.],
                    "nodes": n,
                    "senders": [],
                    "receivers": []
            }
            edges = []
            for k1 in range(len(n)-1):
                for k2 in range(k1+1, len(n)):
                    edges.append([0.])
                    graph["senders"].append(k1)
                    graph["receivers"].append(k2)
            graph["edges"] = np.array(edges).astype(np.float32)
            graph_dicts.append(graph)
            labels.append(y[i])
            print(graph["edges"].shape)

        return graph_dicts, labels


    


def main(_):
    with open('data/train.pkl', 'rb') as f:
        graphs = pickle.load(f)

    train_graphs = graphs[:5120]
    test_graphs = graphs[5120:]

    #  create placeholder
    if FLAGS.model == "rnn":
        pos = tf.placeholder(tf.float32, [None, None, 3])
        ids = tf.placeholder(tf.int32, [None, None])
        lattice = tf.placeholder(tf.float32, [None, 3, 3])
        y = tf.placeholder(tf.int64, [None])
        seq_len = tf.placeholder(tf.int32, [None])
        h_hat = model.naive(pos, ids, seq_len)

    elif FLAGS.model == "graph":
        input_graph = utils_tf.placeholders_from_data_dicts(modified_graphs[0:1])
        y = tf.placeholder(tf.int64, [None])
        lattice = tf.placeholder(tf.float32, [None, 3, 3])
        h_hat = model.tinet(input_graph)

    # merge lattice information w/ atoms
    h_lattice = tf.reshape(lattice, [-1, 9])
    h = tf.concat([h_hat, h_lattice], axis=1)
    y_hat = tf.layers.dense(h, 4, activation=None)

    # count total params
    N = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('Total number of params is ', N)

    # defind loss and metric
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
    loss = tf.reduce_mean(loss)
    predicts = tf.argmax(y_hat, 1)
    correct_pred = tf.equal(predicts, y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # optimizer
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    init = tf.global_variables_initializer()          

    # saver
    saver = tf.train.Saver()

    # run session
    with tf.Session() as sess:
        sess.run(init)

        for i in range(100000):
            sample_idx = np.random.choice(5120, [1])
            batch_graphs = []
            for k in sample_idx:
                batch_graphs.append(train_graphs[k])

            if FLAGS.model == 'rnn':
                batch_pos, batch_ids, batch_lattice, batch_y, batch_seq_len = build_dict(batch_graphs, FLAGS.model)
                feed_dict = {pos: batch_pos, ids: batch_ids, 
                        lattice: batch_lattice, y:batch_y, seq_len:batch_seq_len}
            elif FLAGS.model == "graph":
                raise

            loss_value, _ = sess.run(
                [loss, optimizer], feed_dict=feed_dict)
  
            if i % 100 == 99:
                total_loss = 0
                total_acc = 0
                discrimiate = np.zeros((4,4)).astype(np.int32)
                for j in range(len(test_graphs)):
                    if FLAGS.model == 'rnn':
                        batch_pos, batch_ids, batch_lattice, batch_y, batch_seq_len = build_dict([test_graphs[j]], FLAGS.model)
                        feed_dict = {pos: batch_pos, ids: batch_ids, 
                                lattice: batch_lattice, y:batch_y, seq_len:batch_seq_len}
                    elif FLAGS.model == "graph":
                        raise
                    # if FLAGS.model == "rnn":
                    #     feed_dict = {x: [test_x[j]], c: [test_c[j]], y: [test_y[j]], g: [test_g[j]]}
                    # elif FLAGS.model == "graph":
                    #     test_graph_data = utils_np.data_dicts_to_graphs_tuple([test_graph[j]])
                    #     feed_dict = {input_graph: test_graph_data, y: [test_label[j]]}

                    loss_value, acc, predicts_value = sess.run(
                                    [loss, accuracy, predicts], feed_dict=feed_dict)
                    total_acc += acc
                    total_loss += loss_value
                    discrimiate[test_graphs[j]['y']][predicts_value[0]] += 1
                
                print('Test Loss is ', total_loss / len(test_graphs), '; accuracy is ', total_acc / len(test_graphs))
                print(discrimiate)
    
        save_path = saver.save(sess, "model/model.ckpt")
        print("Model saved in path: %s" % save_path)

if __name__ == '__main__':
    tf.app.run()