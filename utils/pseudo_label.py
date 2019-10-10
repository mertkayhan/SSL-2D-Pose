import tensorflow as tf
import sys; sys.path.append("../")
import argparse
from src.u_net import create_conv_net as unet
from src.get_batch import Batch
import numpy as np
import os


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sess = tf.Session()

    net_in = tf.placeholder(tf.float32, shape=[1, 256, 320, 3])
    is_training = tf.placeholder(tf.bool)
    keep_prob = .9
    num_parts = 5
    num_connections = 4

    net_out, _ = unet(net_in, keep_prob, 3, num_parts + num_connections,
                      is_training=is_training, features_root=64, alpha=1 / 5.5)

    restore_op, restore_dict = tf.contrib.framework.assign_from_checkpoint(
        args.model_dir + "/model.ckpt",
        tf.contrib.slim.get_variables_to_restore(),
        ignore_missing_vars=True
    )
    sess.run(restore_op, feed_dict=restore_dict)
    print("Restored session and reset global step")

    b = Batch(args.root, 1, testing=True, augment=False, dataset="ENDOVIS", include_unlabelled=True, pseudo_label=True)
    max_it = len(b.unlabelled_img_list)

    print("Found", max_it, "unlabeled images.")

    for i in range(max_it):
        img, label, _, _ = b.get_batch()
        output = sess.run(net_out, feed_dict={net_in: img, is_training: False})
        assert output[0].shape[0] == 256
        assert output[0].shape[1] == 320
        assert output[0].shape[2] == 9
        np.save(os.path.join(args.output_dir, b.name_list[0] + ".npy"), output[0])

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Directory that contains the data", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the pseudo labels")
    args = parser.parse_args()

    main(args)