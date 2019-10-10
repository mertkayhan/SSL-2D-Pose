import sys; sys.path.append("../")
from src.img_utils import read, augment
import numpy as np
from glob import glob
from random import shuffle
import json
import cv2


ENDOVIS_SHAPE = (576, 720)
ENDOVIS_TARGET_SHAPE = (320, 256)
RMIT_SHAPE = (480, 640)
RMIT_TARGET_SHAPE = (384, 288)


class Batch(object):

    def __init__(self, root, batch_size, testing=False, augment=True, dataset="ENDOVIS", include_unlabelled=True,
                 pseudo_label=False, train_postprocessing=False):
        if dataset == "RMIT":
            mode = "test" if testing else "training"
            img_dir = root + "/" + mode + "/image"
            label_dir = root + "/" + mode + "/label"
            attention_dir = root + "/attention_maps"
            self.img_list = glob(img_dir + "/*png")
            self.unlabelled_img_list = None
        elif dataset == "ENDOVIS":
            img_dir = root + "/labelled_test/" if testing else root + "/labelled_train"
            if not use_partial_labels and not train_postprocessing:
                label_dir = root + "/test_labels/" if testing else root + "/training_labels"
            elif train_postprocessing:
                label_dir = root + "/test_labels_postprocessing/" if testing else root + "/training_labels_postprocessing"
            self.img_list = glob(img_dir + "/*jpg")
            if include_unlabelled and (not testing or pseudo_label):
                unlabelled_img_dirs = root + "/unlabelled_train"
                self.unlabelled_img_list = glob(unlabelled_img_dirs + "/*")
                print(len(self.unlabelled_img_list))
            if pseudo_label:
                self.pseudo_dir = root + "/pseudo_labels"
            else:
                self.unlabelled_img_list = None
        else:
            raise ValueError
        assert(len(self.img_list) > 0)
        shuffle(self.img_list)
        self.batch_size = batch_size
        self.idx = 0
        self.pseudo_idx = 0
        self.label_dir = label_dir
        # self.attention_dir = attention_dir
        self.name_list = []
        self.org_shapes = []
        self.testing = testing
        self.augment = augment
        self.real_shape = ENDOVIS_SHAPE if dataset == "ENDOVIS" else RMIT_SHAPE
        self.target_shape = ENDOVIS_TARGET_SHAPE if dataset == "ENDOVIS" else RMIT_TARGET_SHAPE
        self.num_parts = 4 if dataset == "RMIT" else None
        if self.num_parts is None:
            if train_postprocessing:
                self.num_parts = 10
            else:
                self.num_parts = 9
        if dataset == "ENDOVIS":
            with open(root + "/instrument_count.json", "r") as p:
                self.instrument_count = json.load(p)
        else:
            self.instrument_count = {}
        self.batch_instrument_count = None
        self.pseudo_label = pseudo_label
        # print(self.instrument_count.keys())

    def get_batch(self):

        img_dir_list = []
        target_dir_list = []
        attention_dir_list = []
        self.name_list = []
        self.org_shapes = []
        self.batch_instrument_count = []

        # collect directories

        N = self.batch_size if self.unlabelled_img_list is None else int(self.batch_size * .8)
        
        for i in range(N):
            img_dir_list.append(self.img_list[self.idx])
            fname = self.img_list[self.idx].split("/")[-1]
            fname = fname.split(".")[0]
            self.name_list.append(fname)
            target_dir_list.append(self.label_dir + "/" + fname + ".npy")
            try:
                self.batch_instrument_count.append(self.instrument_count[fname + ".npy"])
            except:
                self.batch_instrument_count.append(1)
            # attention_dir_list.append(self.attention_dir + "/" + fname + ".png")
            self.idx += 1
            if self.idx == len(self.img_list):
                self.idx = 0
                shuffle(self.img_list)

        for i in range(self.batch_size - N):
            if self.pseudo_label:
                id = self.pseudo_idx
            else:
                id = np.random.randint(0, len(self.unlabelled_img_list))
            img_dir_list.append(self.unlabelled_img_list[id])
            fname = self.unlabelled_img_list[id].split("/")[-1]
            fname = fname.split(".")[0]
            self.name_list.append(fname)
            if self.pseudo_label:
                target_dir_list.append(self.pseudo_dir + "/" + fname + ".npy")
            else:
                target_dir_list.append(None)
            if self.pseudo_idx + 1 == len(self.unlabelled_img_list) and self.pseudo_label:
                self.pseudo_idx = 0
                shuffle(self.unlabelled_img_list)
            else:
                self.pseudo_idx += 1

        img_batch = get_img_batch(img_dir_list)
        target_batch = get_label_batch(target_dir_list, (self.real_shape[0], self.real_shape[1], self.num_parts))
        # attention_batch = get_attention_batch(attention_dir_list)

        img_patches = []
        target_patches = []
        attention_patches = []

        # process batch
        for i in range(self.batch_size):
            resized_p = cv2.resize(img_batch[i], self.target_shape)
            resized_t = cv2.resize(target_batch[i], self.target_shape)
            resized_a = np.zeros_like(resized_p)
            if not self.testing and self.augment:
                resized_p, resized_t, resized_a = augment(resized_p, resized_t, resized_a)
            # rescale pixels between [-1, 1]
            resized_p -= 127.5
            resized_p /= 127.5
            # add noise to avoid overfitting
            resized_t += np.random.uniform(low=-.01, high=.01)
            # print(resized_a.min())
            img_patches.append(resized_p)
            target_patches.append(resized_t)
            attention_patches.append(resized_a)

        # z = attention_patches[0]
        # cv2.imshow("z", z)
        # x = target_patches[0] * 255
        # y = (img_patches[0] * 127.5) + 127.5
        # cv2.imshow("y", y[:, :, 3].astype("uint8"))
        # cv2.imshow("y", y[:, :, (2, 1, 0)].astype("uint8"))
        # cv2.imshow("x1", x[:, :, 0].astype("uint8"))
        # cv2.imshow("x2", x[:, :, 1].astype("uint8"))
        # cv2.imshow("x3", x[:, :, 2].astype("uint8"))
        # print("Number of instruments", self.batch_instrument_count[0])
        # cv2.imshow("x4", x[:, :, 3].astype("uint8"))
        # cv2.imshow("x5", x[:, :, 4].astype("uint8"))
        # cv2.imshow("x6", x[:, :, 5].astype("uint8"))
        # cv2.imshow("x7", x[:, :, 6].astype("uint8"))
        # cv2.imshow("x8", x[:, :, 7].astype("uint8"))
        # cv2.imshow("x9", x[:, :, 8].astype("uint8"))
        # cv2.waitKey(0)

        label_mask = [1.] * N + [0.] * (self.batch_size - N) if not self.pseudo_label else [1.] * self.batch_size

        return img_patches, target_patches, attention_patches, label_mask


def get_attention_batch(dir_list):

    tmp = cv2.imread(dir_list[0], cv2.IMREAD_GRAYSCALE)
    batch = np.zeros((len(dir_list), tmp.shape[0], tmp.shape[1], 1), dtype=np.float32)
    batch[0, :, :, :] = tmp.reshape(tmp.shape[0], tmp.shape[1], 1)

    for i, d in enumerate(dir_list[1:]):
        batch[i + 1, :, :, :] = cv2.imread(d, cv2.IMREAD_GRAYSCALE).reshape(tmp.shape[0], tmp.shape[1], 1)

    return batch


def get_img_batch(dir_list):

    tmp = read(dir_list[0])
    batch = np.zeros((len(dir_list), ) + tmp.shape, dtype=np.float32)
    batch[0, :, :, :] = tmp

    for i, d in enumerate(dir_list[1:]):
        batch[i+1, :, :, :] = read(d)

    return batch


def get_label_batch(dir_list, label_dim):

    batch = np.zeros((len(dir_list), ) + label_dim, dtype=np.float32)

    for i, d in enumerate(dir_list):
        try:
            batch[i, :, :, :] = np.load(d)
        except:
            batch[i, :, :, :] = 0.

    return batch


if __name__ == "__main__":
    pass
