import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import model
import graph_nets as gn
import numpy as np
import pickle
import tensorflow as tf


def main():
    with open('data/train.pkl', 'rb') as f:
        atomId, name_data, x_data, y_data, g_data = pickle.load(f)

    input_x = [i[0] for i in x_data]
    input_c = [i[1] for i in x_data]

    x = tf.placeholder(tf.float32, [None, None, 3])
    c = tf.placeholder(tf.int32, [None, None])
    d = len(atomId)

    y_hat = model.naive(x, c, d)

    # load model
    saver = tf.train.Saver()

    ti = []
    tci = []

    # sess predict
    with tf.Session() as sess:
        saver.restore(sess, "model/model.ckpt")

        for i in range(len(x_data)):
            y_value = sess.run(y_hat, feed_dict={x:[input_x[i]], c:[input_c[i]]})
            y_value = np.argmax(y_value)
            print(name_data[i], y_data[i], y_value)
            if y_value == 2:
                ti.append(name_data[i])
            if y_value == 3:
                tci.append(name_data[i])  

    print("TI:", ti)
    print("TCI:", tci)         

if __name__ == '__main__':
    main()