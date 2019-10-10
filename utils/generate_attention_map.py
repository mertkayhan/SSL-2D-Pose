import sys; sys.path.append("../")
import numpy as np
import os
from glob import glob
import argparse
from utils.annotation_parser import parse_annotation
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_dir", help="Directory that contains the labels", type=str, required=True)
    parser.add_argument("--save_dir", help="Directory to save the heatmaps", required=True)
    parser.add_argument("--dataset", default="RMIT")
    args = parser.parse_args()

    if args.dataset != "RMIT":
        raise NotImplementedError

    save_dir = args.save_dir + "/attention_maps"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    label_list = sorted(glob(args.label_dir + "/*txt"), key=lambda x: x[0] < x[1])
    parsed = parse_annotation(label_list)

    for i in range(3):
        data = parsed["seq" + str(i+1)]
        size = len(data["fname"])

        for j in range(size):
            focus_map = np.zeros((480, 640), dtype=np.float32)
            focus_map = cv2.circle(focus_map, (int(data["p2"][j][1]), int(data["p2"][j][0])), 10, (255, 255, 255),
                                   thickness=-1)
            focus_map = cv2.circle(focus_map, (int(data["p3"][j][1]), int(data["p3"][j][0])), 10, (255, 255, 255),
                                   thickness=-1)
            focus_map = cv2.circle(focus_map, (int(data["p4"][j][1]), int(data["p4"][j][0])), 10, (255, 255, 255),
                                   thickness=-1)
            cv2.imwrite(save_dir + "/" + data["fname"][j] + ".png", focus_map)


if __name__ == "__main__":
    main()