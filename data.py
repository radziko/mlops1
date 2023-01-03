import torch
import numpy as np

def mnist():
    # exchange with the corrupted mnist dataset
    trains = [
        np.load("C:/Users/juliu/OneDrive/DTU/dtu_mlops/data/corruptmnist/train_0.npz", mmap_mode="r"),
        np.load("C:/Users/juliu/OneDrive/DTU/dtu_mlops/data/corruptmnist/train_1.npz", mmap_mode="r"),
        np.load("C:/Users/juliu/OneDrive/DTU/dtu_mlops/data/corruptmnist/train_2.npz", mmap_mode="r"),
        np.load("C:/Users/juliu/OneDrive/DTU/dtu_mlops/data/corruptmnist/train_3.npz", mmap_mode="r"),
        np.load("C:/Users/juliu/OneDrive/DTU/dtu_mlops/data/corruptmnist/train_4.npz", mmap_mode="r")
    ]

    images, labels = [], []
    for x in trains:
        images.append(x["images"])
        labels.append(x["labels"])

    train = [np.concatenate(images), np.concatenate(labels)]
    test_set = np.load("C:/Users/juliu/OneDrive/DTU/dtu_mlops/data/corruptmnist//test.npz", mmap_mode="r")
    test = [test_set["images"], test_set["labels"]]
    return train, test
