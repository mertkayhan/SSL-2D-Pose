import tensorflow as tf
import numpy as np
# import sys; sys.path.append("../")
from get_batch import Batch
from u_net import create_conv_net as unet
import argparse
import os
import cv2
from vat import virtual_adversarial_loss


def train(input_t, output_map, alpha, max_it, root, batch_size, is_training, id, use_vat, use_pseudo_labels,
          use_mean_teacher, dataset):
    """
    :param input_t: input tensor
    :param output_map: output layer of the network
    :param alpha: placeholder for leaky relu
    :param max_it: maximum training iterations
    :param root: base directory that contains the images
    :param batch_size: batch size
    :param is_training: toggle training
    :param id: GPU id
    :param use_vat: Enable VAT
    :param use_pseudo_labels: Use pseudo labels
    :param use_mean_teacher: Use mean teacher
    :param dataset: Choose dataset
    :return:
    """

    h = 256 if dataset == "ENDOVIS" else 288
    w = 320 if dataset == "ENDOVIS" else 384
    num_parts = 5 if dataset == "ENDOVIS" else 4
    num_connections = 4 if dataset == "ENDOVIS" else 0

    # GPU Config
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.95)

    # Set up placeholders
    y = tf.placeholder(tf.float32, shape=[None, h, w, num_parts + num_connections])
    lr = tf.placeholder(tf.float32)
    loss_mask = tf.placeholder(tf.float32, shape=[batch_size])

    # Loss
    if not use_mean_teacher:
        avr_loss = tf.losses.mean_squared_error(y, output_map,
                                                weights=tf.reshape(loss_mask,
                                                                   [batch_size, 1, 1, 1]))
    if use_mean_teacher:
        ema = tf.train.ExponentialMovingAverage(decay=.95)

        def ema_getter(getter, name, *args, **kwargs):
            var = getter(name, *args, **kwargs)
            ema_var = ema.average(var)
            return ema_var if ema_var else var

        tf.get_variable_scope().set_custom_getter(ema_getter)
        model_vars = tf.trainable_variables()
        output_student = output_map
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema.apply(model_vars))
        output_teacher, _ = unet(input_t, .9 if dataset == "RMIT" else .7, 3,
                                 num_parts + num_connections,
                                 is_training=is_training,
                                 features_root=64,
                                 alpha=alpha)
        output_teacher = tf.stop_gradient(output_teacher)
        avr_loss = batch_size / tf.reduce_sum(loss_mask) * \
                   tf.losses.mean_squared_error(y, output_student,
                                                weights=tf.reshape(loss_mask,
                                                                   [batch_size, 1, 1, 1]))
        m = tf.placeholder(tf.float32, shape=[])
        avr_loss = avr_loss + m * .1 * tf.losses.mean_squared_error(output_teacher, output_student)

    if use_vat:
        avr_loss = batch_size / tf.reduce_sum(loss_mask) * avr_loss + \
                   virtual_adversarial_loss(input_t, y, is_training=is_training, alpha=alpha)

    # Adam solver
    with tf.variable_scope("Adam", reuse=tf.AUTO_REUSE):
        opt = tf.train.AdamOptimizer(lr).minimize(avr_loss)

    # Start session and initialize weights
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                            allow_soft_placement=True,
                                            log_device_placement=True))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000)

    b_train = Batch(root, batch_size, dataset="ENDOVIS",
                    include_unlabelled=use_vat or use_mean_teacher or use_tvm,
                    pseudo_label=use_pseudo_labels)
    b_test = Batch(root, batch_size, dataset="ENDOVIS", include_unlabelled=False, testing=True, augment=False,
                   train_postprocessing=False)

    current_lr = 1e-3 
    print("Chosen lr:", current_lr)

    # if model_dir is not None:
    #     restore_op, restore_dict = tf.contrib.framework.assign_from_checkpoint(
    #         model_dir + "/model.ckpt",
    #         tf.contrib.slim.get_variables_to_restore(),
    #         ignore_missing_vars=True
    #     )
    #     sess.run(restore_op, feed_dict=restore_dict)
    #     print("Restored session")

    # save graph
    writer = tf.summary.FileWriter(logdir='logdir', graph=sess.graph)
    writer.flush()

    if use_vat:
        test_interval = 250
    else:
        test_interval = 200

    def sigmoid_schedule(global_step, warm_up_steps=20000):
        if global_step > warm_up_steps:
            return 1.

        return np.exp(-5. * (1. - (global_step / warm_up_steps)) ** 2)

    for i in range(max_it):

        imgs, targets, _, mask = b_train.get_batch()

        current_loss, net_out, _ = sess.run(
            [avr_loss, output_map, opt],
            feed_dict={input_t: imgs,
                       y: targets,
                       lr: current_lr,
                       is_training: True,
                       alpha: 1 / np.random.uniform(low=3, high=8),
                       loss_mask: mask,
                       m: sigmoid_schedule(i)
                       }
        )

        if i % 100 == 0:
            print("Current regression loss:", current_loss.sum())
            loc_pred = []
            loc_true = []
            for ch in range(num_parts):
                if b_train.batch_instrument_count[0] == 1:
                    _, _, _, m_loc1 = cv2.minMaxLoc(net_out[0, :, :, ch])
                    loc_pred.append(m_loc1)
                    _, _, _, m_loc2 = cv2.minMaxLoc(targets[0][:, :, ch])
                    loc_true.append(m_loc2)
                else:
                    pass

            print("For the first sample-> Predicted: {}    Ground Truth: {}\n".format(loc_pred, loc_true))

        # save model for evaluation
        if i % test_interval == 0 and i != 0:

            print("Testing at iteration", i, "...")
            dir2save = os.path.join("tmp" + str(i), "model.ckpt")
            save_path = saver.save(sess, dir2save)
            print("Saved model to", save_path)

    sess.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Directory that contains the data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--gpu_id", type=str, default="1", help="Select a gpu")
    parser.add_argument("--use_vat", type=int, default=0, help="Enables VAT")
    parser.add_argument("--use_pseudo_labels", type=int, default=0, help="Enables pseudo-label usage")
    parser.add_argument("--use_mean_teacher", type=int, default=0, help="Enables mean teacher")
    parser.add_argument("--dataset", type=str, default="ENDOVIS", help="Choose RMIT or Endovis to train on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    dataset = args.dataset.upper()

    # init network
    ch = 3
    h = 256 if dataset == "ENDOVIS" else 288
    w = 320 if dataset == "ENDOVIS" else 384
    x = tf.placeholder(tf.float32, shape=[args.batch_size, h, w, ch])
    is_training = tf.placeholder(tf.bool)
    alpha = tf.placeholder_with_default(1 / 5.5, [], name="alpha_lrelu")
    num_parts = 5 if dataset == "ENDOVIS" else 4
    num_connections = 4 if dataset == "ENDOVIS" else 0
    keep_prob = .9 if dataset == "RMIT" else .7
    output_map, _ = unet(x, keep_prob, ch,
                         num_parts + num_connections,
                         is_training=is_training,
                         features_root=64,
                         alpha=alpha)

    train(x, output_map, alpha, 50000, args.root, args.batch_size,
          is_training, args.gpu_id, args.use_vat, 
          args.use_pseudo_labels, args.use_mean_teacher, args.dataset)


if __name__ == "__main__":
    main()
