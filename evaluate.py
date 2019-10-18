import tensorflow as tf
import cv2
# import sys; sys.path.append("../")
from get_batch import Batch
import numpy as np
import argparse
import json
import os
from nms import nms
from line_integral import compute_integral
from scipy.optimize import linear_sum_assignment
from u_net import create_conv_net as unet


def evaluate_ENDOVIS(root, model):

    T = 20  # threshold

    # GPU Config
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.9)

    # Start session and initialize weights
    # tf.reset_default_graph()
    # imported_meta = tf.train.import_meta_graph(model + "/model.ckpt.meta")
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # imported_meta.restore(sess, tf.train.latest_checkpoint(model + "/"))
    # for n in tf.get_default_graph().as_graph_def().node:
    #     if "output_map" in n.name:
    #         print(n.name)

    net_in = tf.placeholder(tf.float32, shape=[1, 256, 320, 3])
    y = tf.placeholder(tf.float32, shape=[1, 256, 320, 9])
    is_training = tf.placeholder(tf.bool)
    keep_prob = .9
    num_parts = 5
    num_connections = 4

    net_out, attention = unet(net_in, keep_prob, 3, num_parts + num_connections,
                              is_training=is_training, features_root=64, alpha=1/5.5)  
    tv = tf.image.total_variation(net_out)
    loss = tf.losses.mean_squared_error(labels=y, predictions=net_out)
    # net_out = post_processing(net_out)
    # print(attention[0].get_shape())
    # print(attention[1].get_shape())

    if model is not None:
        restore_op, restore_dict = tf.contrib.framework.assign_from_checkpoint(
            model + "/model.ckpt",
            tf.contrib.slim.get_variables_to_restore(),
            ignore_missing_vars=True
        )
        sess.run(restore_op, feed_dict=restore_dict)
        print("Restored session and reset global step")

    # net_in = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
    # is_training = tf.get_default_graph().get_tensor_by_name("Placeholder_1:0")
    # try:
    #     net_out = tf.get_default_graph().get_tensor_by_name("output_map/Relu:0")
    # except:
    # net_out = tf.get_default_graph().get_tensor_by_name("output_map/conv2d/dropout/cond/Merge:0")

    mode = ("training", "test")
    testing = (False, True)

    for m in range(2):

        print("Results for", mode[m])

        b = Batch(root, 1, testing=testing[m], augment=False, dataset="ENDOVIS", include_unlabelled=False,
                  train_postprocessing=True)  # False for MSE
        false_pos = np.zeros((5, ), dtype=np.float32)
        false_neg = np.zeros((5, ), dtype=np.float32)
        true_pos = np.zeros((5, ), dtype=np.float32)
        rmse = np.zeros_like(true_pos)
        mae = np.zeros_like(rmse)
        counter = np.zeros_like(rmse)

        precision = lambda fp, tp: tp / (tp + fp)
        recall = lambda fn, tp: tp / (tp + fn)
        # exclude = ("test5", "test6")
        # exclude = ("test1", "test2", "test3", "test4")
        exclude = ()
        w_multiplier = 720. / 320.
        h_multiplier = 576. / 256.
        avr_loss = 0.

        for _ in range(len(b.img_list)):
            img, label, _, _ = b.get_batch()
            # if b.batch_instrument_count[0] == 1:
            #     continue
            skip = False
            for e in exclude:
                if e in b.name_list[0]:
                    skip = True
                    break
            if skip:
                continue
            # t_loss = sess.run(loss, feed_dict={net_in: img, is_training: False, y: label})
            # avr_loss += t_loss

            
            output, a1, a0, total_var = sess.run([net_out, attention[1], attention[0], tv],
                                                 feed_dict={net_in: img, is_training: False})
            # print(total_var)
            blur = output[0].copy()
            blur[:, :, :5] = cv2.GaussianBlur(blur[:, :, :5], (T+1, T+1), 0)
            _, blur[:, :, :5] = nms(blur[:, :, :5])

            # if blur[:, :, 5:].std() < .01:
            if 1000 > total_var > 700:
                mask = cv2.addWeighted(blur[:, :, 5:], 1, cv2.GaussianBlur(blur[:, :, 5:], (T+1, T+1), 0), -1, 0)
                blur[:, :, 5:] += mask

            loc_pred = [[], [], [], [], []]
            loc_true = [[], [], [], [], []]

            k = 5
            for i in range(5):
                heatmap = blur[:, :, i].copy()
                for j in range(k):
                    _, _, _, max_loc = cv2.minMaxLoc(heatmap)
                    if max_loc[0] == max_loc[1] == 0:
                        break
                    loc_pred[i].append(max_loc)
                    y, x = max_loc
                    heatmap[x-5:x+5, y-5:y+5] = 0.

            for ch in range(10):
                _, _, _, max_loc = cv2.minMaxLoc(label[0][:, :, ch])
                if max_loc[0] != 0 and max_loc[1] != 0:
                    loc_true[ch % 5].append(max_loc)

            # print(loc_pred[0])
            # print(loc_true[0])
            # return

            candidates = [[], [], [], []]

            for idx, tmp in enumerate([(0, 2, 5), (1, 2, 6), (2, 3, 7), (3, 4, 8)]):

                joint_idx1, joint_idx2, connection_idx = tmp[0], tmp[1], tmp[2]

                matching_scores = np.zeros((len(loc_pred[joint_idx1]), len(loc_pred[joint_idx2])), dtype=np.float32)

                for y, pt1 in enumerate(loc_pred[joint_idx1]):
                    for x, pt2 in enumerate(loc_pred[joint_idx2]):

                        matching_scores[y, x] = compute_integral(pt1, pt2, blur[:, :, connection_idx])
                        # print("left2head", matching_scores)

                # print(matching_scores)
                row_idx, col_idx = linear_sum_assignment(-matching_scores)

                for a, c in zip(row_idx, col_idx):
                    candidates[idx].append((loc_pred[joint_idx1][a], loc_pred[joint_idx2][c]))

            # print(candidates)

            parsed = []
            for pairs in candidates[-1]:
                shaft, end = pairs
                for next_pairs in candidates[-2]:
                    head, shaft_next = next_pairs
                    if shaft[0] == shaft_next[0] and shaft[1] == shaft_next[1]:
                        parsed.append([head, shaft, end])

            # print("parsed top:", parsed2)

            for i, partial_pose in enumerate(parsed):
                head, _, _ = partial_pose
                for next_pairs in candidates[-3]:
                    right, head_next = next_pairs
                    if head[0] == head_next[0] and head[1] == head_next[1]:
                        parsed[i].insert(0, right)
                for next_pairs in candidates[-4]:
                    left, head_next = next_pairs
                    if head[0] == head_next[0] and head[1] == head_next[1]:
                        parsed[i].insert(0, left)
            # print(parsed)

            for i, pose in enumerate(parsed):
                if len(pose) < 5:
                    for _ in range(5 - len(pose)):
                        parsed[i].insert(0, ())

            parse_failed = False
            final_prediction = [[], [], [], [], []]
            if len(parsed) == 2:
                inst1, inst2 = parsed
                final_prediction = list(zip(inst1, inst2))
            elif len(parsed) == 1:
                final_prediction = parsed
            else:
                parse_failed = True
                for i, pair in enumerate(candidates):
                    # print(pair[0][0])
                    if len(pair) >= 2:
                        final_prediction[i] = [pair[0][0], pair[0][1]]
                    elif len(pair) == 1:
                        final_prediction[i] = [pair[0][0]]
                    else:
                        final_prediction[i] = []
                # print(final_prediction, "\n")

            # print(final_prediction)
            # print(loc_true)

            # return

            for k in range(5):
                try:
                    cost_matrix = np.zeros((len(loc_true[k]), len(final_prediction[k])), dtype=np.float32)
                    pred = final_prediction[k]
                except IndexError:
                    # print(final_prediction[0])
                    cost_matrix = np.zeros((len(loc_true[k]), len(final_prediction[0][k])), dtype=np.float32)
                    pred = [final_prediction[0][k]]

                # print(b.batch_instrument_count[0] - len(final_prediction[k]))
                for y, e_true in enumerate(loc_true[k]):
                    for x, e_pred in enumerate(pred):  # (final_prediction[k]):
                        # print(e_true, e_pred)
                        try:
                            cost_matrix[y, x] = ((e_pred[0] - e_true[0]) * h_multiplier) ** 2 + \
                                                ((e_pred[1] - e_true[1]) * w_multiplier) ** 2
                        except IndexError:
                            if len(final_prediction[0][k]) != 0:
                                cost_matrix[y, x] = 10000
                            else:
                                continue

                row_idx, col_idx = linear_sum_assignment(cost_matrix)

                if len(pred) < b.batch_instrument_count[0]:
                    false_neg[k] += abs(len(pred) - b.batch_instrument_count[0])

                for r, c in zip(row_idx, col_idx):
                    # check true positive and additional false positive
                    mae[k] += np.sqrt(cost_matrix[r, c])
                    counter[k] += 1
                    if np.sqrt(cost_matrix[r, c]) < T:
                        rmse[k] += cost_matrix[r, c]
                        true_pos[k] += 1
                        # print(cost_matrix[r, c])
                    else:
                        false_pos[k] += 1
                        
        f1 = lambda p, r: (2 * p * r) / (p + r)
        p = precision(false_pos, true_pos)
        r = recall(false_neg, true_pos)
        print(true_pos, false_neg, false_pos)
        print("RMSE", np.sqrt(rmse / true_pos))
        print("Precision", p)
        print("Recall", r)
        print("F1", f1(p, r))
        print("MEA", mae / counter)
        print("\n")
        
        # print(avr_loss / (float(len(b.img_list))))

def evaluate_RMIT(root, model):
    T = 15  # threshold

    # GPU Config
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.9)

    # Start session and initialize weights
    tf.reset_default_graph()
    imported_meta = tf.train.import_meta_graph(model + "/model.ckpt.meta")
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    imported_meta.restore(sess, tf.train.latest_checkpoint(model + "/"))
    # for n in tf.get_default_graph().as_graph_def().node:
    #     if "output_map" in n.name:
    #         print(n.name)

    net_in = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
    is_training = tf.get_default_graph().get_tensor_by_name("Placeholder_1:0")
    try:
        net_out = tf.get_default_graph().get_tensor_by_name("output_map/Relu:0")
    except:
        net_out = tf.get_default_graph().get_tensor_by_name("output_map/conv2d/dropout/cond/Merge:0")

    mode = ("training", "test")
    testing = (False, True)

    for k in range(2):

        print("Results for", mode[k])

        diff_acc = np.zeros((4, 2), dtype=np.float32)
        unscaled = np.zeros_like(diff_acc)
        b = Batch(root, 1, testing=testing[k], augment=False, dataset="RMIT")
        detected = [0, 0, 0, 0]
        euc_list = []
        failed = []

        for i in range(len(b.img_list)):
            img, label, _, _ = b.get_batch()
            output = sess.run(net_out, feed_dict={net_in: [img[0]] * 5, is_training: False})
            blur = cv2.GaussianBlur(output[0], (5, 5), 0)
            loc_pred = get_minmaxloc(blur)
            loc_true = get_minmaxloc(label[0])
            w = 640
            h = 480
            ########################################################
            d = np.abs(loc_pred - loc_true)
            d[:, 0] = (d[:, 0] * h) / 288.
            d[:, 1] = (d[:, 1] * w) / 384.
            euc = []
            for l in range(4):
                euc_dist = np.sqrt(d[l, 0] ** 2 + d[l, 1] ** 2)
                if euc_dist < T:
                    detected[l] += 1
                elif l == 2:
                    failed.append(img[0])
                    # print("Failed:", b.name_list[0])
                euc.append(euc_dist)
            euc_list.append(euc)
            ########################################################
            diff = np.sqrt((loc_true - loc_pred) ** 2)
            unscaled += diff
            diff[:, 0] = (diff[:, 0] * h) / 288.
            diff[:, 1] = (diff[:, 1] * w) / 384.
            diff_acc += diff

            img_ = img[0].copy() * 127.5 + 127.5
            img_ = img_.astype("uint8")
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
            for j, c in enumerate(loc_pred):
                img_ = cv2.circle(img_, (int(c[0]), int(c[1])), 2, colors[j], -1)
            cv2.imwrite("output_frames/{}.png".format(b.name_list[0]), img_)

        diff_acc /= float(len(b.img_list))
        unscaled /= float(len(b.img_list))
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        print("Scaled:", diff_acc)
        print("Unscaled:", unscaled)
        print("Accuracy:", np.array(detected) / float(len(b.img_list)))
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        with open("evaluation_results.json", "w") as f:
            json.dump({"p1": "(" + str(diff_acc[0, 0]) + ", " + str(diff_acc[0, 1]) + ")",
                       "p2": "(" + str(diff_acc[1, 0]) + ", " + str(diff_acc[1, 1]) + ")",
                       "p3": "(" + str(diff_acc[2, 0]) + ", " + str(diff_acc[2, 1]) + ")"},
                      f)

        counter = [0, 0, 0, 0]
        acc = [0, 0, 0, 0]

        for x in euc_list:
            for j in range(4):
                if x[j] < 15:
                    counter[j] += 1
                    acc[j] += x[j] ** 2

        acc = np.array(acc)
        acc = np.divide(acc, counter)
        acc = np.sqrt(acc)
        print("\n RMSE on detected parts:")
        print(acc)

    sess.close()

    return failed


def get_minmaxloc(M):

    loc = []

    for ch in range(M.shape[-1]):
        _, _, _, max_loc = cv2.minMaxLoc(M[:, :, ch])
        loc.append(max_loc)

    return np.array(loc, dtype=np.float32)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Directory that contains the data", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--gpu_id", type=str, default="0", help="Gpu id")
    parser.add_argument("--dataset", required=True, help="Name of the dataset")
    parser.add_argument("--speedtest", required=False, help="Measure runtime", type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.dataset == "RMIT":
        failed = evaluate_RMIT(args.root, args.model_dir)
    elif args.dataset == "ENDOVIS":
        evaluate_ENDOVIS(args.root, args.model_dir)
    else:
        raise ValueError

    """
    for img in failed:
        img = img * 127.5 + 127.5
        cv2.imshow("failed", img[:, :, (2, 1, 0)].astype("uint8"))
        cv2.waitKey(0)
    """


if __name__ == "__main__":
    main()



