import argparse
import os
from glob import glob
import subprocess as sp
import sys; sys.path.append("../")
from utils.annotation_parser import parse_annotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", help="Directory that contains the images", type=str, required=True)
    parser.add_argument("--label_dir", help="Directory that contains the annotations", type=str, required=True)
    parser.add_argument("--dataset", help="Splits depend on the dataset", type=str, default="RMIT")
    parser.add_argument("--save_dir", help="Directory to save the training/test files", required=True)
    parser.add_argument("--heatmap_dir", help="Directory that contains the heatmaps", required=True)
    args = parser.parse_args()

    if args.dataset == "RMIT":
        RMIT(args)
    elif args.dataset == "ENDOVIS":
        ENDOVIS(args)
    else:
        raise NotImplementedError


def ENDOVIS(args):
    pass


def RMIT(args):
    label_list = sorted(glob(args.label_dir + "/*txt"), key=lambda x: x[0] < x[1])
    parsed = parse_annotation(label_list)

    training_dir = args.save_dir + "/training"
    test_dir = args.save_dir + "/test"

    if not os.path.exists(training_dir):
        os.mkdir(training_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    training_img_dir = training_dir + "/image"
    training_label_dir = training_dir + "/label"
    test_img_dir = test_dir + "/image"
    test_label_dir = test_dir + "/label"

    if not os.path.exists(training_img_dir):
        os.mkdir(training_img_dir)
    if not os.path.exists(training_label_dir):
        os.mkdir(training_label_dir)
    if not os.path.exists(test_img_dir):
        os.mkdir(test_img_dir)
    if not os.path.exists(test_label_dir):
        os.mkdir(test_label_dir)

    for i in range(3):
        size = len(parsed["seq{}".format(i + 1)]["fname"])
        train_size = size // 2 + 1  # size of training split

        for j in range(train_size):
            fname = parsed["seq{}".format(i + 1)]["fname"][j]
            sp.call("cp {} {}".format(args.img_dir + "/" + fname + ".png", training_img_dir + "/" + fname + ".png"),
                    shell=True)
            sp.call("cp {} {}".format(args.heatmap_dir + "/" + fname + ".npy",
                                      training_label_dir + "/" + fname + ".npy"),
                    shell=True)

        for j in range(train_size, size, 1):
            fname = parsed["seq{}".format(i + 1)]["fname"][j]
            sp.call("cp {} {}".format(args.img_dir + "/" + fname + ".png", test_img_dir + "/" + fname + ".png"),
                    shell=True)
            sp.call("cp {} {}".format(args.heatmap_dir + "/" + fname + ".npy",
                                      test_label_dir + "/" + fname + ".npy"),
                    shell=True)


if __name__ == "__main__":
    main()