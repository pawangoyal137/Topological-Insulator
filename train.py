import os
import model
import graph_nets as gn
import numpy as np
import pickle
import tensorflow as tf
import random

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf
from itertools import permutations
import random
tf.app.flags.DEFINE_string(
    'model', 'rnn', 'The name of the RNN model:graph, rnn')
tf.app.flags.DEFINE_integer(
    'RNN_num_layers', 2, 'Number of layers for RNN model')
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
            pos_empty = np.zeros((max_len - len(i['atoms']), 3))
            this_pos = np.array(i['coords'])
            this_pos = np.concatenate([this_pos, pos_empty], axis=0)
            pos.append(this_pos)

            ats_empty = np.zeros((max_len - len(i['atoms'])))
            this_ats = np.array(i['atoms'])
            this_ats = np.concatenate([this_ats, ats_empty], axis=0)
            ids.append(this_ats)

            seq_len.append(len(i['atoms']))
        return pos, ids, lattice, y, seq_len

    elif model == "graph":
        graph_dicts = []
        labels = []
        lattices = []
        for i in graphs:
            N = len(i['atoms'])

            if len(i['atoms']) <= 1:
                continue

            nodes = np.array(i['atoms']).astype(np.int32)
            graph = {
                "globals": np.reshape(i['lattice'], [9]).astype(np.float32),
                "nodes": nodes,
                "senders": [],
                "receivers": [],
                "edges": []
            }
            edges = []

            for k1 in range(N - 1):
                for k2 in range(k1 + 1, N):
                    a = i['coords'][k1]
                    b = i['coords'][k2]
                    edges.append(a - b)
                    graph["senders"].append(k1)
                    graph["receivers"].append(k2)
            graph["edges"] = np.array(edges).astype(np.float32)
            graph_dicts.append(graph)

            labels.append(i['y'])
            lattices.append(i['lattice'])
        return graph_dicts, labels, lattices


def main(_):
    with open('data/train.pkl', 'rb') as f:
        graphs = pickle.load(f)
    f.close()

    ## merge TI and TCI

    for g in graphs:
        if g['y'] == 3:
            g['y'] = 2

    train_graphs = graphs[:5120]
    test_graphs = graphs[5120:]

    ## for sampling ##
    category_ids = [[] for i in range(3)]
    for i in range(len(train_graphs)):
        c = train_graphs[i]['y']
        category_ids[c].append(i)

    #  create placeholder
    if FLAGS.model == "rnn":
        pos = tf.placeholder(tf.float32, [None, None, 3])
        ids = tf.placeholder(tf.int32, [None, None])
        lattice = tf.placeholder(tf.float32, [None, 3, 3])
        y = tf.placeholder(tf.int64, [None])
        seq_len = tf.placeholder(tf.int32, [None])
        h_hat = model.naive(pos, ids, seq_len, FLAGS.RNN_num_layers)

    elif FLAGS.model == "graph":
        modified_graphs, _, _ = build_dict(train_graphs, 'graph')
        input_graph = utils_tf.placeholders_from_data_dicts(
            modified_graphs[0:1])
        y = tf.placeholder(tf.int64, [None])
        lattice = tf.placeholder(tf.float32, [None, 3, 3])
        h_hat = model.tinet(input_graph)

    # merge lattice information w/ atoms
    h_lattice = tf.reshape(lattice, [-1, 9])
    h = tf.concat([h_hat, h_lattice], axis=1)
    y_hat = tf.layers.dense(h, 3, activation=None)

    # count total params
    N = np.sum([np.prod(v.get_shape().as_list())
                for v in tf.trainable_variables()])
    print('Total number of params is ', N)

    # define loss and metric
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=y_hat)
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
        for i in range(10000):

            batch_graphs = []
            for k in range(32):
                cat = int(np.random.random() * 3)
                sample_idx = random.choice(category_ids[cat])
                batch_graphs.append(train_graphs[sample_idx])

            if FLAGS.model == 'rnn':
                batch_pos, batch_ids, batch_lattice, batch_y, batch_seq_len = build_dict(
                    batch_graphs, FLAGS.model)

                # TODO 
                # @Pawan: you are still inserting arrays to tensor. 
                # And it is still very slow
                # batch_length,seq_length,_=np.shape(batch_pos)
                # batch_pos_new=np.array([]).reshape(0,seq_length,3)
                # batch_ids_new=np.array([]).reshape(0,seq_length)
                # for i in range(batch_length):
                #     actual_len=batch_seq_len[i]
                #     batch_pos_i=batch_pos[i][:actual_len]
                #     batch_ids_i=batch_ids[i][:actual_len].reshape(1,-1)
                #     concatenated_matrix=np.concatenate((batch_pos_i,batch_ids_i.T),axis=1)
                #     np.random.shuffle(concatenated_matrix)
                #     pos_empty = np.zeros((seq_length - actual_len, 4))
                #     this_pos=np.concatenate((concatenated_matrix,pos_empty),axis=0)
                #     batch_pos_i=this_pos[:,:-1]
                #     batch_pos_i=batch_pos_i.reshape(1,seq_length,-1)
                #     batch_ids_i=this_pos[:,-1:].T
                #     batch_pos_new=np.concatenate((batch_pos_new,batch_pos_i),axis=0)
                #     batch_ids_new=np.concatenate((batch_ids_new,batch_ids_i),axis=0)
                
                # batch_pos=batch_pos_new
                # batch_ids=batch_ids_new
                feed_dict = {
                    pos: batch_pos,
                    ids: batch_ids,
                    lattice: batch_lattice,
                    y: batch_y,
                    seq_len: batch_seq_len}
            elif FLAGS.model == "graph":
                batch_graphnets, batch_labels, batch_lattice = build_dict(
                    batch_graphs, FLAGS.model)
                train_batch_graph_data = utils_np.data_dicts_to_graphs_tuple(
                    batch_graphnets)
                feed_dict = {
                    input_graph: train_batch_graph_data,
                    y: batch_labels,
                    lattice: batch_lattice}
            loss_value, _ = sess.run(
                [loss, optimizer], feed_dict=feed_dict)

            if i % 100 == 99:
                total_loss = 0
                total_acc = 0

                relevant_elements = [0, 0, 0]
                selected_elements = [0, 0, 0]
                true_positives = [0, 0, 0]

                discrimiate = np.zeros((3, 3)).astype(np.int32)
                for j in range(len(test_graphs)):
                    if FLAGS.model == 'rnn':
                        batch_pos, batch_ids, batch_lattice, batch_y, batch_seq_len = build_dict(
                            [test_graphs[j]], FLAGS.model)
                        feed_dict = {
                            pos: batch_pos,
                            ids: batch_ids,
                            lattice: batch_lattice,
                            y: batch_y,
                            seq_len: batch_seq_len}
                    elif FLAGS.model == "graph":
                        batch_graphnets, batch_labels, batch_lattice = build_dict(
                            [test_graphs[j]], FLAGS.model)
                        if not batch_labels:
                            continue
                        test_batch_graph_data = utils_np.data_dicts_to_graphs_tuple(
                            batch_graphnets)
                        feed_dict = {
                            input_graph: test_batch_graph_data,
                            y: batch_labels,
                            lattice: batch_lattice}
                        
                    loss_value, acc, predicts_value = sess.run(
                        [loss, accuracy, predicts], feed_dict=feed_dict)

                    total_acc += acc
                    total_loss += loss_value
                    discrimiate[test_graphs[j]['y']][predicts_value[0]] += 1

                    selected_elements[predicts_value[0]] += 1
                    relevant_elements[test_graphs[j]['y']] += 1
                    if predicts_value[0] == test_graphs[j]['y']:
                        true_positives[predicts_value[0]] += 1

                precision = None
                recall = None
                if selected_elements[2] != 0:
                    precision = true_positives[2] / selected_elements[2] 

                if relevant_elements[2] != 0:
                    recall = true_positives[2] / relevant_elements[2]

                if precision is not None and recall is not None:
                    f1_score = 2 * precision * recall / (precision + recall)

                print('F1 score is', f1_score)

                print(
                    'Test Loss is ',
                    total_loss /
                    len(test_graphs),
                    '; accuracy is ',
                    total_acc /
                    len(test_graphs))
                print(discrimiate)

        save_path = saver.save(sess, "model/" + FLAGS.model + "/model.ckpt")
        print("Model saved in path: %s" % save_path)


if __name__ == '__main__':
    tf.app.run()
