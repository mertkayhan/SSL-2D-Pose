import tensorflow as tf
# import sys; sys.path.append("../")
from u_net import create_conv_net

epsilon = .1
num_power_iterations = 1
xi = 1e-6


def entropy_min(logit):
    p = tf.nn.softmax(logit)
    return -tf.reduce_mean(tf.reduce_sum(p * logsoftmax(logit), [1, 2]))


def logsoftmax(x):
    xdev = x - tf.reduce_max(x, [1, 2], keepdims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), [1, 2], keepdims=True))
    return lsm


def get_normalized_vector(d):
    # print(d.get_shape())
    # d /= (1e-12 + tf.reduce_max(tf.abs(d), range(1, len(d.get_shape())), keep_dims=True))
    d /= (1e-12 + tf.reduce_max(tf.abs(d), axis=[1, 2], keepdims=True))
    # d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), range(1, len(d.get_shape())), keep_dims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), axis=[1, 2], keepdims=True))

    return d


def generate_virtual_adversarial_perturbation(x, logit, is_training=True, alpha=None):
    d = tf.random_normal(shape=tf.shape(x))

    for _ in range(num_power_iterations):
        d = xi * get_normalized_vector(d)
        logit_p = logit
        logit_m, _ = create_conv_net(x + d, .9, 3, 9, is_training=is_training, features_root=64, alpha=alpha)
        dist = tf.losses.mean_squared_error(logit_p, logit_m)
        grad = tf.gradients(dist, [d], aggregation_method=2)[0]
        d = tf.stop_gradient(grad)

        return epsilon * get_normalized_vector(d)


def virtual_adversarial_loss(x, logit, is_training=True, name="vat_loss", alpha=None):
    r_vadv = generate_virtual_adversarial_perturbation(x, logit, is_training=is_training, alpha=alpha, dist_=dist,
                                                       bs=bs)
    logit = tf.stop_gradient(logit)
    logit_p = logit
    logit_m, _ = create_conv_net(x + r_vadv, .9, 3, 9, is_training=is_training, features_root=64, alpha=alpha)
    loss = tf.losses.mean_squared_error(logit_p, logit_m)
    return tf.identity(loss, name=name)
