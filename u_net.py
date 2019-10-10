import tensorflow as tf
from collections import OrderedDict
import numpy as np

"""I took the code from 'https://github.com/jakeret/tf_unet' and modified it."""


def create_conv_net(x, keep_prob, channels, n_class, layers=3, features_root=16, filter_size=3, pool_size=2,
                    is_training=None, alpha=None):

    with tf.variable_scope("dau-net", reuse=tf.AUTO_REUSE):

        def rlrelu(x):
            return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

        in_node = x
        pools = OrderedDict()
        deconv = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()

        # in_size = 1000
        # size = in_size

        counter = 1

        # down layers
        for layer in range(0, layers):
            with tf.name_scope("down_conv_{}".format(str(layer))):
                features = 2 ** layer * features_root
                stddev = np.sqrt(2 / (filter_size ** 2 * features))
                if layer == 0:
                    w1 = weight_variable([filter_size, filter_size, channels, features], stddev, name="w1")
                else:
                    w1 = weight_variable([filter_size, filter_size, features // 2, features], stddev, name="w1")
                b1 = bias_variable([features], name="b1")

                w2 = weight_variable([filter_size, filter_size, features, features], stddev, name="w2")
                b2 = bias_variable([features], name="b2")

                conv1 = conv2d(in_node, w1, b1, keep_prob, is_training)
                conv1 = tf.contrib.layers.group_norm(conv1, mean_close_to_zero=True)
                tmp_h_conv = rlrelu(conv1)

                conv2 = conv2d(tmp_h_conv, w2, b2, keep_prob, is_training)
                conv2 = tf.contrib.layers.group_norm(conv2, mean_close_to_zero=True)
                dw_h_convs[layer] = rlrelu(conv2)

                # size -= 2 * 2 * (filter_size // 2) # valid conv

                if layer < layers - 1:
                    pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                    in_node = pools[layer]
                    # size /= pool_size

        in_node = dw_h_convs[layers - 1]
        attention_list = []

        # up layers
        for layer in range(layers - 2, -1, -1):
            with tf.name_scope("up_conv_{}".format(str(layer))):
                features = 2 ** (layer + 1) * features_root
                stddev = np.sqrt(2 / (filter_size ** 2 * features))

                wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")
                bd = bias_variable([features // 2], name="bd")
                h_deconv = rlrelu(deconv2d(in_node, wd, pool_size) + bd)
                h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)

                with tf.name_scope("attention{}".format(layer)):
                    attention = tf.contrib.layers.conv2d(h_deconv_concat, features_root, 3, activation_fn=rlrelu)
                    attention = tf.layers.dropout(attention, rate=1 - keep_prob, training=is_training)
                    attention = tf.contrib.layers.conv2d(attention, int(round(np.sqrt(features_root))), 3,
                                                         activation_fn=rlrelu)
                    attention = tf.layers.dropout(attention, rate=1 - keep_prob, training=is_training)
                    attention = tf.contrib.layers.conv2d(attention, 1, 1, activation_fn=tf.nn.sigmoid)
                    attention_list.append(attention)
                    h_deconv_concat = tf.multiply(h_deconv_concat, attention)
                    
                deconv[layer] = h_deconv_concat

                w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev, name="w1")
                w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2")
                b1 = bias_variable([features // 2], name="b1")
                b2 = bias_variable([features // 2], name="b2")

                conv1 = conv2d(h_deconv_concat, w1, b1, keep_prob, is_training)
                conv1 = tf.contrib.layers.group_norm(conv1, mean_close_to_zero=True)
                h_conv = rlrelu(conv1)

                conv2 = conv2d(h_conv, w2, b2, keep_prob, is_training)
                conv2 = tf.contrib.layers.group_norm(conv2, mean_close_to_zero=True)
                in_node = rlrelu(conv2)

                up_h_convs[layer] = in_node

                # size *= pool_size
                # size -= 2 * 2 * (filter_size // 2)  # valid conv

        # Output Map
        with tf.name_scope("output_map"):

            weight = weight_variable([1, 1, features_root, n_class], stddev)
            bias = bias_variable([n_class], name="bias")
            output_map = conv2d(in_node, weight, bias, tf.constant(1.0), is_training)
            up_h_convs["out"] = output_map

    return up_h_convs["out"], attention_list


def weight_variable(shape, stddev=0.1, name="weight"):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def weight_variable_devonc(shape, stddev=0.1, name="weight_devonc"):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)


def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, b, keep_prob_, is_training, stride=1):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        return tf.layers.dropout(conv_2d_b, rate=1-keep_prob_, training=is_training)


def deconv2d(x, W,stride):
    with tf.name_scope("deconv2d"):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID',
                                      name="conv2d_transpose")


def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')


def crop_and_concat(x1,x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)


if __name__ == "__main__":
    pass
