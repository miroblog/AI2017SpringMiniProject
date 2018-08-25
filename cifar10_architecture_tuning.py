from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from datetime import datetime
import data_helpers
LOGDIR = '/tmp/17springAI/cifar10/architecture/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'

IMAGE_PIXELS = 3072
CLASSES = 10

data_sets = data_helpers.load_data()

def activation(act_func, logit):
    if act_func == "relu":
        return tf.nn.relu(logit)
    else:
        return tf.nn.sigmoid(logit)
# logit
def logits(input, size_in, size_out, name="logits"):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    logit = (tf.matmul(input, w) + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("logits", logit)
    return logit, w, b


# fully conected layer
def fc_layer(input, size_in, size_out, act_func, name="fc"):
    with tf.name_scope(name):
        logit, w, b = logits(input, size_in, size_out, name="logits")
        act = activation(act_func, logit)
        tf.summary.histogram("activations", act)
        return act


# batch layer
def batch_logits(input, size_in, size_out, act_func):
    # you don't need biases when batch normalizing
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    # logits ; weighted sum
    z1_BN = tf.matmul(input, w)
    bmean1, bvar1 = tf.nn.moments(z1_BN, [0])
    beta1 = tf.Variable(tf.zeros([size_out]), name="batch_b")
    epsilon = 1e-3
    # BN_logit = tf.nn.batch_normalization(z1_BN, bmean1, bvar1, beta1, scale1, epsilon)
    if(act_func == "relu"):
        # when applying batch normalization to reLu, we don't need scaling factor
        BN_logit = tf.nn.batch_normalization(z1_BN, bmean1, bvar1, beta1, None, epsilon)
    elif(act_func == "sigmoid"):
        scale1 = tf.Variable(tf.ones([size_out]), name="batch_s")
        BN_logit = tf.nn.batch_normalization(z1_BN, bmean1, bvar1, beta1, scale1, epsilon)
    # when applying batch normalization to reLu, we don't need scaling factor
    # tf.summary.histogram("batch_scale", scale1)
    tf.summary.histogram("batch_beta", beta1)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("BN_logit", BN_logit)
    return BN_logit


# batch layer
def batch_layer(input, size_in, size_out, act_func, name="bl"):
    with tf.name_scope(name):
        BN1 = batch_logits(input, size_in, size_out, act_func)
        act = activation(act_func, BN1)
        tf.summary.histogram("activations", act)
        return act


# runs different model each time, hparam is a string specification for the model
# hpram is also used in the created tensorboard summarye
def cifar10_model(learning_rate, regularization, hparam, dropout_rate, n_hidden_layer, n_hidden_unit, act_func):
    tf.reset_default_graph()
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))

    # input layer
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS], name='images')
    x_image = tf.reshape(x, [-1, 32, 32, 1])
    tf.summary.image('input', x_image, 3)

    # label to compare
    y_ = tf.placeholder(tf.int64, shape=[None], name='image-labels')
    keep_prob = tf.placeholder(tf.float32)
    y = tf.one_hot(y_,10,1.0,0.0,-1)

    layers = []
    if regularization == "drop_out":
        for i in range(n_hidden_layer):
            if i == 0:
                layers.insert(i, tf.nn.dropout(fc_layer(x, IMAGE_PIXELS, n_hidden_unit, act_func, "h"+str(i+1)), keep_prob))
            else:
                layers.insert(i, tf.nn.dropout(fc_layer(layers[i-1], n_hidden_unit, n_hidden_unit, act_func, "h"+str(i+1)), keep_prob))
        logit, W, B = logits(layers[n_hidden_layer-1], n_hidden_unit, 10)

    elif regularization == 'batch_normalization':
        for i in range(n_hidden_layer):
            if i == 0:
                layers.insert(i, batch_layer(x, IMAGE_PIXELS, n_hidden_unit, act_func, "h"+str(i+1)))
            else:
                layers.insert(i, batch_layer(layers[i-1], n_hidden_unit, n_hidden_unit, act_func, "h"+str(i+1)) )
        logit = batch_logits(layers[n_hidden_layer-1], n_hidden_unit, 10, act_func)

    else:
        for i in range(n_hidden_layer):
            if i == 0:
                layers.insert(i, fc_layer(x, IMAGE_PIXELS, n_hidden_unit, act_func, "h"+str(i+1)) )
            else:
                layers.insert(i, fc_layer(layers[i-1], n_hidden_unit, n_hidden_unit, act_func,"h"+str(i+1)))
        logit, W, B = logits(layers[n_hidden_layer-1], n_hidden_unit, 10)

    ## softmax layer - last layer for classification
    Y = tf.nn.softmax(logit)

    # loss function
    with tf.name_scope("xent"):
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logit, labels=y), name="xent")
        tf.summary.scalar("xent", xent)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    writer_train = tf.summary.FileWriter(LOGDIR + hparam + "_train")
    writer_train.add_graph(sess.graph)
    writer_test = tf.summary.FileWriter(LOGDIR + hparam + "_test")
    writer_test.add_graph(sess.graph)

    num_epochs = 200
    # training accuracy
    list_test_acc = list()

    # Generate input data batches
    zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])
    # batch size : 100, max steps : (steps in a single epoch) * num of epochs
    batches = data_helpers.gen_batch(list(zipped_data), 100 , 500*100*num_epochs)

    for k in range(num_epochs):
        print(str(k) + "th epoch")
        for i in range(500):
            batch = next(batches)
            batch_xs, batch_ys = zip(*batch)
            if i % 100 == 0:
                [train_accuracy, s_train] = sess.run([accuracy, summ],feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1})
                writer_train.add_summary(s_train, k * 500 + i)
                [test_accuracy, s_test] = sess.run([accuracy, summ],feed_dict={x: data_sets['images_test'],y_: data_sets['labels_test'],  keep_prob: 1})
                writer_test.add_summary(s_test, k * 500 + i)
                print('Step {:d}, training accuracy {:g}'.format(k * 500 + i, train_accuracy))
                print('Step {:d}, test accuracy {:g}'.format(k * 500 + i, test_accuracy))
            # dropout_rate will only be used when dropout is enabled
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: dropout_rate})
        test_acc = accuracy.eval(feed_dict={x: data_sets['images_test'],y_: data_sets['labels_test'], keep_prob : 1})
        list_test_acc.append(test_acc)
        # use early stopping
        if k > 10 and np.mean(list_test_acc[-10:-5]) > np.mean(list_test_acc[-5:]):
            print("Seems like it starts to overfit, aborting the training")
            break


def make_hparam_string(act_func, learning_rate, regularization, dropout_rate, n_hidden_layer, n_hidden_unit):
    return "%s,lr_%.0E,%s,dr_%f,hl_%d,hu_%d" % (act_func, learning_rate, regularization, dropout_rate, n_hidden_layer, n_hidden_unit)

def main():
    for act_func in ["sigmoid", "relu"]:
        for learning_rate in [1E-3]:
            for n_hidden_layer in [1, 2, 3]:
                for n_hidden_unit in [10, 50, 100, 200, 400, 800]:
                    for regularization in ["normal", "drop_out", "batch_normalization"]:
                        if regularization == "drop_out":
                            for dropout_rate in [0.5]:
                                hparam = make_hparam_string(act_func, learning_rate, regularization, dropout_rate, n_hidden_layer,
                                                            n_hidden_unit)
                                print('Starting run for %s' % hparam)
                                cifar10_model(learning_rate, regularization, hparam, dropout_rate, n_hidden_layer,
                                            n_hidden_unit, act_func)
                        else:
                            for dropout_rate in [1]:
                                hparam = make_hparam_string(act_func, learning_rate, regularization, dropout_rate, n_hidden_layer,
                                                            n_hidden_unit)
                                print('Starting run for %s' % hparam)
                                cifar10_model(learning_rate, regularization, hparam, dropout_rate, n_hidden_layer,
                                            n_hidden_unit, act_func)

if __name__ == '__main__':
    main()
