import sys; sys.path.append("../")
import tensorflow as tf
from src.u_net import create_conv_net
import os
from time import time


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sess = tf.Session()

    x = tf.ones(shape=[1, 256, 320, 3], dtype=tf.float32)
    net_out, _ = create_conv_net(x, .9, 3, 9, is_training=False, features_root=64, alpha=1/5.5)

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print("Total number of trainable parameters:", total_parameters)

    sess.run(tf.global_variables_initializer())

    begin = time()
    max_it = 1000
    for _ in range(max_it):
        sess.run(net_out)
    end = time()

    print("Elapsed time", (end - begin) / max_it, "ms")


if __name__ == "__main__":
    main()