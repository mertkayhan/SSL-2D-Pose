from glob import glob
import argparse
import pickle
import os


def extract_bbox(annotation, alpha=1):
    relevant = annotation[1:]  # shaft is not relevant
    max_y = max(relevant, key=lambda x: x[1])[1]
    max_x = max(relevant, key=lambda x: x[0])[0]
    min_y = min(relevant, key=lambda x: x[1])[1]
    min_x = min(relevant, key=lambda x: x[0])[0]
    delta = alpha * max([max_x - min_x, max_y - min_y])
    bbox = [(min_y - delta, min_x - delta), (max_y + delta, max_x + delta)]
    return bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Number of joints depend on the dataset", type=str, default="RMIT")
    parser.add_argument("--save_dir", help="Directory to save pickle", required=True)
    parser.add_argument("--root", help="Directory that contains the annotations", required=True)
    args = parser.parse_args()

    if args.dataset != "RMIT":
        raise NotImplementedError

    parsed = {}

    for annotation in glob(args.root + "/*txt"):
        with open(annotation, "r") as f:
            for line in f:
                split = line.split(" ")[:9]
                if split[1] == "-1":
                    continue
                split[1:] = map(int, split[1:])
                parsed[split[0]] = extract_bbox([[split[1], split[2]],
                                                 [split[3], split[4]],
                                                 [split[5], split[6]],
                                                 [split[7], split[8]]])
    save_dir = args.save_dir + "/bbox"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    with open(save_dir + "/bbox.pickle", "wb") as p:
        pickle.dump(parsed, p)


if __name__ == "__main__":
    main()