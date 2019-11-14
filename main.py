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


def build_graph(pos, ids):
    graph_dicts = []
    for i in range(len(pos)):
        graph = {
                "globals": [0.],
                "nodes": np.array(pos[i]).astype(np.float32),
                "edges": [[0.]],
                "senders": [0],
                "receivers": [0],
        }
        graph_dicts.append(graph)
    return graph_dicts



def main(_):
    with open('data/train.pkl', 'rb') as f:
        atomId, name_data, x_data, y_data, g_data = pickle.load(f)

    train_x = [i[0] for i in x_data[:5120]]
    train_c = [i[1] for i in x_data[:5120]]
    train_graph = build_graph(train_x, train_c)
    train_y = y_data[:5120]

    test_x = [i[0] for i in x_data[5120:]]
    test_c = [i[1] for i in x_data[5120:]]
    test_graph = build_graph(test_x, test_c)
    test_y = y_data[5120:]

    d = len(atomId)

    if FLAGS.model == "rnn":

        x = tf.placeholder(tf.float32, [None, None, 3])
        c = tf.placeholder(tf.int32, [None, None])
        y = tf.placeholder(tf.int64, [None])
        y_hat = model.naive(x, c, d)

    elif FLAGS.model == "graph":

        #  create placeholder
        input_graph = utils_tf.placeholders_from_data_dicts(train_graph[0:1])
        y = tf.placeholder(tf.int64, [None])
        y_hat = model.tinet(input_graph)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
    loss = tf.reduce_mean(loss)
    predicts = tf.argmax(y_hat, 1)
    correct_pred = tf.equal(tf.argmax(y_hat, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    init = tf.global_variables_initializer()          

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(100000):
            k = i % 5120

            if FLAGS.model == "rnn":
                feed_dict = {x: [train_x[k]], c: [train_c[k]], y: [train_y[k]]}
            elif FLAGS.model == "graph":
                input_graph_data = utils_np.data_dicts_to_graphs_tuple([train_graph[k]])
                # feed_dict = {input_graph: input_graph_data, y: [train_y[k]]}
                feed_dict = utils_tf.get_feed_dict(
                    input_graph, input_graph_data)
                feed_dict[y] = [train_y[k]]
            loss_value, _ = sess.run(
                [loss, optimizer], feed_dict=feed_dict)
  
            if k % 512 == 0:
                total_loss = 0
                total_acc = 0
                discrimiate = np.zeros((4,4)).astype(np.int32)
                for j in range(len(test_y)):
                    if FLAGS.model == "rnn":
                        feed_dict = {x: [train_x[j]], c: [train_c[j]], y: [train_y[j]]}
                    elif FLAGS.model == "graph":
                        test_graph_data = utils_np.data_dicts_to_graphs_tuple([test_graph[j]])
                        feed_dict = {input_graph: test_graph_data, y: [train_y[j]]}

                    loss_value, acc, predicts_value = sess.run(
                                    [loss, accuracy, predicts], feed_dict=feed_dict)
                    total_acc += acc
                    total_loss += loss_value
                    discrimiate[test_y[j]][predicts_value] += 1
                print('Test Loss is ', total_loss / len(test_y), '; accuracy is ', total_acc / len(test_y))
                print(discrimiate)
    
        save_path = saver.save(sess, "model/model.ckpt")
        print("Model saved in path: %s" % save_path)

if __name__ == '__main__':
    tf.app.run()