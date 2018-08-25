from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from datetime import datetime
import data_helpers

LOGDIR = '/tmp/17springAI/cifar10/objectiveFunc/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'

IMAGE_PIXELS = 3072
CLASSES = 10

data_sets = data_helpers.load_data()

def activation(act_func, logit):
    if act_func == "relu":
        return tf.nn.relu(logit)
    else:
        return tf.nn.sigmoid(logit)

def logits(input, size_in, size_out):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    logit = (tf.matmul(input, w) + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("logits", logit)
    return logit, w, b


# fully conected layer
def fc_layer(input, size_in, size_out, act_func,  name="fc"):
    with tf.name_scope(name):
        logit, w, b = logits(input, size_in, size_out)
        act = activation(act_func, logit)
        tf.summary.histogram("activations", act)
        return act, w, b


def cifar10_model(learning_rate, objectiveFunc, hparam, act_func):
    tf.reset_default_graph()
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4)))

    # Define input placeholders
    # images_placeholder - x
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS], name='images')
    x_image = tf.reshape(x, [-1, 32, 32, 1])
    tf.summary.image('input', x_image, 3)

    # labels_placeholder - y_
    y_ = tf.placeholder(tf.int64, shape=[None], name='image-labels')
    keep_prob = tf.placeholder(tf.float32)

    y = tf.one_hot(y_, 10, 1.0, 0.0, -1)

    h1, W1, B1 = fc_layer(x, IMAGE_PIXELS, 100, act_func, "h1")
    logit, W2, B2 = logits(h1, 100, 10)
    Y = tf.nn.softmax(logit)

    ## changing loss function
    if objectiveFunc == "mean_sq_err":
        with tf.name_scope("mean_sq_err"):
            mean_sq_err = tf.reduce_mean(tf.contrib.keras.losses.mean_squared_error(Y, y))
            tf.summary.scalar("mean_sq_err", mean_sq_err)
            loss = mean_sq_err
    elif objectiveFunc == "L2_norm":
        with tf.name_scope("L2_norm"):
            xent = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=logit, labels=y), name="xent")
            L2_lambda = 0.05
            L2_norm = xent + \
                      L2_lambda * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(B1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(B2))
            tf.summary.scalar("L2_norm", L2_norm)
            loss = L2_norm
    else:
        with tf.name_scope("xent"):
            xent = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=logit, labels=y), name="xent")
            tf.summary.scalar("xent", xent)
            loss = xent

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

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
    # batch size 400, max steps 2000
    batches = data_helpers.gen_batch(list(zipped_data), 100, 500 * 100 * num_epochs)

    for k in range(num_epochs):
        print(str(k) + "th epoch")
        for i in range(500):
            batch = next(batches)
            batch_xs, batch_ys = zip(*batch)
            feed_dict = {
                x: batch_xs,
                y_: batch_ys
            }
            if i % 100 == 0:
                [train_accuracy, s_train] = sess.run([accuracy, summ], feed_dict=feed_dict)
                writer_train.add_summary(s_train, k * 500 + i)
                [test_accuracy, s_test] = sess.run([accuracy, summ], feed_dict={x: data_sets['images_test'],
                                                                                y_: data_sets['labels_test']})
                writer_test.add_summary(s_test, k * 500 + i)
                print("train accuracy: " + str(train_accuracy))
                print("test accuracy: " + str(test_accuracy))
            sess.run(train_step, feed_dict=feed_dict)
        test_acc = accuracy.eval(feed_dict={x: data_sets['images_test'], y_: data_sets['labels_test']})
        list_test_acc.append(test_acc)
        if k > 10 and np.mean(list_test_acc[-10:-5]) > np.mean(list_test_acc[-5:]):
            print("Seems like it starts to overfit, aborting the training")
            break


def make_hparam_string(act_func, learning_rate, objective):
    return "%s,lr_%.0E,%s" % (act_func, learning_rate, objective)


def main():
    for act_func in ["sigmoid", "relu"]:
        # You can try adding some more learning rates
        for learning_rate in [0.001]:
            for objective in ["xent", "mean_sq_err", "L2_norm"]:
                # Construct a hyperparameter string for each one
                # def mnist_model(learning_rate, regularization, hparam):
                hparam = make_hparam_string(act_func, learning_rate, objective)
                print('Starting run for %s' % hparam)

                # Actually run with the new settings
                cifar10_model(learning_rate, objective, hparam, act_func)

if __name__ == '__main__':
    main()
