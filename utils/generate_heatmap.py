import sys; sys.path.append("../")
import numpy as np
import os
from glob import glob
import argparse
import json
from utils.annotation_parser import parse_annotation
from utils.heatmap import eval_gaussian, eval_line
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_dir", help="Directory that contains the labels", type=str, required=True)
    parser.add_argument("--save_dir", help="Directory to save the heatmaps", required=True)
    parser.add_argument("--dataset", type=str, default="RMIT")
    parser.add_argument("--post_processing_labels", type=int, default=0,
                        help="Generate labels for post-processing module")
    args = parser.parse_args()

    if args.dataset == "RMIT":
        RMIT(args)
    elif args.dataset == "ENDOVIS":
        ENDOVIS(args)


def ENDOVIS(args):
    def get_line(pt1, pt2):
        # (x, y)
        m = float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])
        b = pt2[1] - m * pt2[0]
        return m, b

    num_classes = 5
    num_connections = 4

    read_folder_name = ("train_labels", "test_labels")
    save_folder_name = ("training_labels_postprocessing", "test_labels_postprocessing")
    num_instruments = {}

    for i in range(2):

        label_dir = os.path.join(args.label_dir, read_folder_name[i])
        json_list = glob(os.path.join(label_dir, "*.json"))

        mapper = {"LeftClasperPoint": 0,
                  "RightClasperPoint": 1,
                  "ShaftPoint": 3,
                  "EndPoint": 4,
                  "HeadPoint": 2}

        for json_ in json_list:
            with open(json_, "r") as j:
                dict_list = json.loads(j.read())
                seq_id = json_.split("/")[-1]
                seq_id = seq_id.split("_")[0]
                for element in dict_list:
                    if element["annotations"] == []:
                        continue
                    if not args.post_processing_labels:
                        heatmap = np.zeros((576, 720, num_classes+num_connections), dtype=np.float32)
                        counter = 0
                        part_coord = {"LeftClasperPoint": [],
                                      "RightClasperPoint": [],
                                      "ShaftPoint": [],
                                      "EndPoint": [],
                                      "HeadPoint": []}
                        for e in element["annotations"]:
                            if e["class"] in mapper.keys():
                                idx = mapper[e["class"]]
                                part_coord[e["class"]].append((e["x"], e["y"]))
                                heatmap[:, :, idx] += eval_gaussian([e["y"], e["x"]], h=576, w=720)
                                counter += 1
                        fname = element["filename"].split("/")[-1]
                        fname = fname.split(".")[0] + "_" + seq_id + ".npy"
                        num_instruments[fname] = counter // num_classes
                        # print(fname)
                        # return

                        # determine the connections
                        for idx in range(counter // num_classes):
                            # print(part_coord)
                            # print("\n")

                            # left2head
                            m, b = get_line(part_coord["LeftClasperPoint"][idx], part_coord["HeadPoint"][idx])
                            heatmap[:, :, 5] += eval_line(part_coord["LeftClasperPoint"][idx], part_coord["HeadPoint"][idx],
                                                          m, b)
                            # rigth2head
                            m, b = get_line(part_coord["RightClasperPoint"][idx], part_coord["HeadPoint"][idx])
                            heatmap[:, :, 6] += eval_line(part_coord["RightClasperPoint"][idx], part_coord["HeadPoint"][idx],
                                                          m, b)
                            # head2shaft
                            m, b = get_line(part_coord["HeadPoint"][idx], part_coord["ShaftPoint"][idx])
                            heatmap[:, :, 7] += eval_line(part_coord["HeadPoint"][idx], part_coord["ShaftPoint"][idx], m, b)
                            # shaft2end
                            m, b = get_line(part_coord["ShaftPoint"][idx], part_coord["EndPoint"][idx])
                            heatmap[:, :, 8] += eval_line(part_coord["ShaftPoint"][idx], part_coord["EndPoint"][idx], m, b)
                    else:
                        heatmap = np.zeros((576, 720, 10), dtype=np.float32)

                        mapper = {"tool1": {"LeftClasperPoint": 0,
                                            "RightClasperPoint": 1,
                                            "ShaftPoint": 3,
                                            "EndPoint": 4,
                                            "HeadPoint": 2
                                            },
                                  "tool2": {"LeftClasperPoint": 5,
                                            "RightClasperPoint": 6,
                                            "ShaftPoint": 8,
                                            "EndPoint": 9,
                                            "HeadPoint": 7
                                            },
                                  "tool4": {"LeftClasperPoint": 5,
                                            "RightClasperPoint": 6,
                                            "ShaftPoint": 8,
                                            "EndPoint": 9,
                                            "HeadPoint": 7
                                            }
                                  }

                        for e in element["annotations"]:
                            try:
                                if e["class"] in mapper[e["id"]].keys():
                                    idx = mapper[e["id"]][e["class"]]
                                    heatmap[:, :, idx] += eval_gaussian([e["y"], e["x"]], h=576, w=720)
                                    # import cv2
                                    # cv2.imshow("heatmap", (heatmap[:, :, idx] * 255).astype("uint8"))
                                    # cv2.waitKey(0)
                            except KeyError as err:
                                print(err)
                                print(e)
                                if e["x"] >= 720//2:
                                    tool_id = "tool1"
                                    idx = mapper[tool_id][e["class"]]
                                    heatmap[:, :, idx] += eval_gaussian([e["y"], e["x"]], h=576, w=720)
                                else:
                                    raise ValueError
                        fname = element["filename"].split("/")[-1]
                        fname = fname.split(".")[0] + "_" + seq_id + ".npy"

                    np.save(os.path.join(args.save_dir, save_folder_name[i], fname), heatmap)

    with open(os.path.join(args.save_dir, "instrument_count.json"), "w") as d:
        json.dump(num_instruments, d)



def RMIT(args):
    save_dir = args.save_dir + "/heatmaps"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    label_list = sorted(glob(args.label_dir + "/*txt"), key=lambda x: x[0] < x[1])
    parsed = parse_annotation(label_list)

    for i in range(3):
        data = parsed["seq" + str(i + 1)]
        size = len(data["fname"])

        for j in range(size):
            heatmap = np.zeros((480, 640, 4), dtype=np.float32)
            heatmap[:, :, 0] = eval_gaussian(data["p1"][j])
            heatmap[:, :, 1] = eval_gaussian(data["p2"][j])
            heatmap[:, :, 2] = eval_gaussian(data["p3"][j])
            heatmap[:, :, 3] = eval_gaussian(data["p4"][j])
            np.save(save_dir + "/" + data["fname"][j] + ".npy", heatmap)


if __name__ == "__main__":
    main()