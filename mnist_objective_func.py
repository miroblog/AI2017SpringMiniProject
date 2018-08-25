import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

LOGDIR = '/tmp/17springAI/mnist/objectiveFunc/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'

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
def fc_layer(input, size_in, size_out,act_func, name="fc" ):
    with tf.name_scope(name):
        logit, w, b = logits(input, size_in, size_out)
        act = activation(act_func, logit)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act, w, b

# runs different model each time, hparam is a string specification for the model
# hpram is also used in the created tensorboard summary
def mnist_model(learning_rate, objectiveFunc, hparam, act_func):
    tf.reset_default_graph()
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4)))

    # input layer
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # to view images on tensorboard
    tf.summary.image('input', x_image, 3)

    # label to compare
    y_ = tf.placeholder(tf.float32, [None, 10], name="labels")
    keep_prob = tf.placeholder(tf.float32)

    h1, W1, B1 = fc_layer(x, 784, 100, act_func, "h1")
    logit, W2, B2 = logits(h1, 100, 10)
    Y = tf.nn.softmax(logit)

    ## changing loss function
    if objectiveFunc == "mean_sq_err":
        with tf.name_scope("mean_sq_err"):
            mean_sq_err = tf.reduce_mean(tf.contrib.keras.losses.mean_squared_error(Y, y_))
            tf.summary.scalar("mean_sq_err", mean_sq_err)
            loss = mean_sq_err
    elif objectiveFunc == "L2_norm":
        with tf.name_scope("L2_norm"):
            xent = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=logit, labels=y_), name="xent")
            L2_lambda = 0.05
            L2_norm = xent + \
                      L2_lambda * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(B1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(B2))
            tf.summary.scalar("L2_norm", L2_norm)
            loss = L2_norm
    else:
        with tf.name_scope("xent"):
            xent = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=logit, labels=y_), name="xent")
            tf.summary.scalar("xent", xent)
            loss = xent


    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y_, 1))
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
            if i % 100 == 0:
                batch_xs, batch_ys = mnist.train.next_batch(100)
                [train_accuracy, s_train] = sess.run([accuracy, summ],
                                                     feed_dict={x: batch_xs, y_: batch_ys})
                writer_train.add_summary(s_train, k * 550 + i)
                [test_accuracy, s_test] = sess.run([accuracy, summ],
                                                   feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                writer_test.add_summary(s_test, k * 550 + i)
                print('Step {:d}, training accuracy {:g}'.format(k * 550 + i, train_accuracy))
                print('Step {:d}, test accuracy {:g}'.format(k * 550 + i, test_accuracy))
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        vacc = accuracy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1})
        list_vacc.append(vacc)
        if k > 10 and np.mean(list_vacc[-10:-5]) > np.mean(list_vacc[-5:]):
            print("Seems like it starts to overfit, aborting the training")
            break

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)


def make_hparam_string(act_func, learning_rate, objective):
    return "%s,lr_%.0E,%s" % (act_func, learning_rate, objective)

def main():
    for act_func in ["sigmoid", "relu"]:
        # You can try adding some more learning rates
        for learning_rate in [1E-4]:
            # Include "False" as a value to try different model architectures:
            for objective in ["xent", "mean_sq_err", "L2_norm"]:
                # def mnist_model(learning_rate, regularization, hparam):
                hparam = make_hparam_string(act_func, learning_rate, objective)
                print('Starting run for %s' % hparam)

                # Actually run with the new settings
                mnist_model(learning_rate, objective, hparam, act_func)

if __name__ == '__main__':
    main()
