import os
import model
import graph_nets as gn
import numpy as np
import pickle
import tensorflow as tf

from train import build_dict

def main():
    with open('data/predict.pkl', 'rb') as f:
        predict_graphs = pickle.load(f)

    
    pos = tf.placeholder(tf.float32, [None, None, 3])
    ids = tf.placeholder(tf.int32, [None, None])
    lattice = tf.placeholder(tf.float32, [None, 3, 3])
    y = tf.placeholder(tf.int64, [None])
    seq_len = tf.placeholder(tf.int32, [None])
    h_hat = model.naive(pos, ids, seq_len)
    h_lattice = tf.reshape(lattice, [-1, 9])
    h = tf.concat([h_hat, h_lattice], axis=1)
    y_hat = tf.layers.dense(h, 4, activation=None)

    # load model
    saver = tf.train.Saver()

    ti = []
    tci = []

    # sess predict
    with tf.Session() as sess:
        saver.restore(sess, "model/model.ckpt")

        for i in range(len(predict_graphs)):
            batch_pos, batch_ids, batch_lattice, batch_y, batch_seq_len = build_dict([predict_graphs[i]], 'rnn')
            feed_dict = {pos: batch_pos, ids: batch_ids, 
                                lattice: batch_lattice, y:batch_y, seq_len:batch_seq_len}
            y_hat_value = sess.run(y_hat, feed_dict=feed_dict)
            y_value = np.argmax(y_hat_value)
            # print(predict_graphs[i]['name'], predict_graphs[i]['y'], y_value)
            if y_value == 3:
                ti.append((y_hat_value[3], predict_graphs[i]['name']))
            if y_value == 2:
                tci.append((y_hat_value[2], predict_graphs[i]['name']))

    ti.sort(reverse=True)            
    tci.sort(reverse=True)            
    print(ti)
    print(tci)

if __name__ == '__main__':
    main()