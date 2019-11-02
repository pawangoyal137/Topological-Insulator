import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import model
import graph_nets as gn
import numpy as np
import pickle
import tensorflow as tf


def main():
    with open('data/data.pkl', 'rb') as f:
        atomId, x_data, y_data = pickle.load(f)

    train_x = [i[0] for i in x_data[:5120]]
    train_c = [i[1] for i in x_data[:5120]]
    train_y = y_data[:5120]
    test_x = [i[0] for i in x_data[5120:]]
    test_c = [i[1] for i in x_data[5120:]]
    test_y = y_data[5120:]

    d = len(atomId)

    x = tf.placeholder(tf.float32, [None, None, 3])
    c = tf.placeholder(tf.int32, [None, None])
    y = tf.placeholder(tf.int64, [None])

    y_hat = model.naive(x, c, d)
    print(y_hat)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
    loss = tf.reduce_mean(loss)
    predicts = tf.argmax(y_hat, 1)
    correct_pred = tf.equal(tf.argmax(y_hat, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    init = tf.global_variables_initializer()          

    with tf.Session() as sess:
        sess.run(init)

        for i in range(100000):
            k = i % 5120
            feed_dict = {x: [train_x[k]], c: [train_c[k]], y: [train_y[k]]}
            loss_value, _ = sess.run(
                [loss, optimizer], feed_dict=feed_dict)
  
            if k % 512 == 0:
                total_loss = 0
                total_acc = 0
                discrimiate = np.zeros((4,4)).astype(np.int32)
                for j in range(len(test_y)):
                    feed_dict = {x: [test_x[j]], c: [test_c[j]], y: [test_y[j]]}
                    loss_value, acc, predicts_value = sess.run(
                                    [loss, accuracy, predicts], feed_dict=feed_dict)
                    total_acc += acc
                    total_loss += loss_value
                    discrimiate[test_y[j]][predicts_value] += 1
                print('Test Loss is ', total_loss / len(test_y), '; accuracy is ', total_acc / len(test_y))
                print(discrimiate)
    
if __name__ == '__main__':
    main()

