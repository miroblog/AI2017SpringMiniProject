import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

LOGDIR = '/tmp/17springAI/mnist/architecture/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'

IMAGE_PIXELS = 784
CLASSES = 10

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
    if(act_func == "relu"):
        # when applying batch normalization to reLu, we don't need scaling factor
        BN_logit = tf.nn.batch_normalization(z1_BN, bmean1, bvar1, beta1, None, epsilon)
    elif(act_func == "sigmoid"):
        scale1 = tf.Variable(tf.ones([size_out]), name="batch_s")
        BN_logit = tf.nn.batch_normalization(z1_BN, bmean1, bvar1, beta1, scale1, epsilon)
        tf.summary.histogram("batch_scale", scale1)
    # when applying batch normalization to reLu, we don't need scaling factor
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

def mnist_model(learning_rate, regularization, hparam, dropout_rate, n_hidden_layer, n_hidden_unit, act_func):
    tf.reset_default_graph()
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))

    # input layer
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS], name="x")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # to view images on tensorboard
    tf.summary.image('input', x_image, 3)

    # label to compare
    y_ = tf.placeholder(tf.float32, [None, 10], name="labels")
    keep_prob = tf.placeholder(tf.float32)

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
                logits=logit, labels=y_), name="xent")
        tf.summary.scalar("xent", xent)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_, 1))
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
    list_vacc = list()

    for k in range(num_epochs):
        print(str(k) + "th epoch")
        for i in range(550):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            if i % 100 == 0:
                [train_accuracy, s_train] = sess.run([accuracy, summ],
                                                     feed_dict={x: batch_xs, y_: batch_ys,
                                                                keep_prob: 1})
                writer_train.add_summary(s_train, k * 550 + i)
                [test_accuracy, s_test] = sess.run([accuracy, summ],
                                                   feed_dict={x: mnist.test.images, y_: mnist.test.labels,
                                                              keep_prob: 1})
                writer_test.add_summary(s_test, k * 550 + i)
                print('Step {:d}, training accuracy {:g}'.format(k * 550 + i, train_accuracy))
                print('Step {:d}, test accuracy {:g}'.format(k * 550 + i, test_accuracy))
            # dropout_rate will only be used when dropout is enabled
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: dropout_rate})

        vacc = accuracy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1})
        list_vacc.append(vacc)
        # use early stopping
        if k > 10 and np.mean(list_vacc[-10:-5]) > np.mean(list_vacc[-5:]):
            print("Seems like it starts to overfit, aborting the training")
            break


def make_hparam_string(act_func, learning_rate, regularization, dropout_rate, n_hidden_layer, n_hidden_unit):
    return "%s,lr_%.0E,%s,dr_%f,hl_%d,hu_%d" % (act_func, learning_rate, regularization, dropout_rate, n_hidden_layer, n_hidden_unit)

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

def main():
    for act_func in ["sigmoid", "relu"]:
        for learning_rate in [1E-3]:
            for n_hidden_layer in [1, 2, 3]:
                for n_hidden_unit in [10, 50, 100, 200, 400, 800]:
                    for regularization in ["normal", "drop_out", "batch_normalization"]:
                        if regularization == "drop_out":
                            for dropout_rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
                                # def mnist_model(learning_rate, regularization, hparam):
                                hparam = make_hparam_string(act_func, learning_rate, regularization, dropout_rate, n_hidden_layer,
                                                            n_hidden_unit)
                                print('Starting run for %s' % hparam)
                                mnist_model(learning_rate, regularization, hparam, dropout_rate, n_hidden_layer,
                                            n_hidden_unit, act_func)
                        else:
                            for dropout_rate in [1]:
                                hparam = make_hparam_string(act_func, learning_rate, regularization, dropout_rate, n_hidden_layer,
                                                            n_hidden_unit)
                                print('Starting run for %s' % hparam)
                                mnist_model(learning_rate, regularization, hparam, dropout_rate, n_hidden_layer,
                                            n_hidden_unit, act_func)

if __name__ == '__main__':
    main()


