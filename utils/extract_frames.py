import cv2
import argparse
import os
from glob import glob


def extract_frames(args):

    if not os.path.exists(args.video_dir) or not os.path.exists(args.save_dir):
        raise ValueError

    video_dirs = glob(os.path.join(args.video_dir, "Dataset*", "*.avi"))
    assert len(video_dirs) != 0

    in_counter = 0
    total_counter = 0

    for path in video_dirs:
        vidObj = cv2.VideoCapture(path)
        # Used as counter variable
        count = 1

        # checks whether frames were extracted
        success = 1

        split_ = path.split("/")

        if split_[-3] == "Training":
            phase = "train"
            label_ext = "training_labels"
        else:
            phase = "test"
            label_ext = "test_labels"

        label_names = glob(os.path.join(args.label_dir, label_ext, "*npy"))
        assert len(label_names) != 0
        label_names = list(map(lambda x: x.split(".")[0], label_names))
        label_names = list(map(lambda x: x.split("/")[-1], label_names))

        while success:
            # vidObj object calls read
            # function extract frames
            success, image = vidObj.read()

            # Saves the frames with frame-count
            if split_[-3] == "Training":
                fname = "img_{0:0=6d}_raw_{1}".format(count, phase + split_[-2][-1])
            else:
                fname = "img_{0:0=4d}_{1}".format(count, phase + split_[-2][-1])

            if fname in label_names:
                in_counter += 1
                save_path = os.path.join(args.save_dir, "labelled_"+phase)
            else:
                save_path = os.path.join(args.save_dir, "unlabelled_"+phase)

            if os.path.exists(save_path):
                cv2.imwrite(os.path.join(save_path, fname+".jpg"), image)
            else:
                raise ValueError(save_path)

            count += 1
            total_counter += 1

    print("Labelled images:", in_counter)
    print("Total images:", total_counter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", help="Directory that contains the videos", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--label_dir", type=str, required=True)
    args = parser.parse_args()

    extract_frames(args)


if __name__ == "__main__":
    main()
