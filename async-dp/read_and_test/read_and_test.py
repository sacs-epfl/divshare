import json
import os
import sys
from datetime import datetime
from os.path import basename, normpath
from pathlib import Path

import torch
from CIFAR10_simple import CIFAR10, LeNet
from torch.nn import CrossEntropyLoss


def eval(base_dir):
    """
    Reads from base_dir/models folder the saved models and evaluates them on the test set of CIFAR10.
    The filenames of the models are of the form: [node id]_model_[iteration]_iter.pt

    For each node the following dictionaries are created:
    1. test_acc: {iteration: test accuracy}
    2. test_loss: {iteration: test loss}
    3. time_test_acc: {time of creation of the model: test accuracy}

    The dictionaries are dumped to a json file named [node id]_testset_results.json

    """

    dataset = CIFAR10()
    model = LeNet()
    loss = CrossEntropyLoss()

    test_eval = {"test_acc": {}, "test_loss": {}}

    model_dir = Path(os.path.join(base_dir, "models"))
    files = os.listdir(model_dir)
    files = [f for f in files if f.endswith(".pt")]

    for file in files:
        print("Now reading ", file)

        model.load_state_dict(torch.load(os.path.join(model_dir, file)))
        model.eval()
        ta, tl = dataset.test(model, loss)

        file_name = file.split("_")
        node_id = file_name[0]
        iteration = file_name[2]

        test_eval["test_acc"][iteration] = ta
        test_eval["test_loss"][iteration] = tl

    # Order the dictionaries by iteration
    test_eval["test_acc"] = {
        k: v
        for k, v in sorted(test_eval["test_acc"].items(), key=lambda item: int(item[0]))
    }
    test_eval["test_loss"] = {
        k: v
        for k, v in sorted(
            test_eval["test_loss"].items(), key=lambda item: int(item[0])
        )
    }

    with open(os.path.join(base_dir, "0_testset_results.json"), "w") as of:
        json.dump(test_eval, of)


if __name__ == "__main__":
    assert len(sys.argv) == 2
    # The args are:
    # 1: path to the folder of the experiment

    eval(sys.argv[1])
