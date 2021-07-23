from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import argparse
import shutil
import numpy as np
import time
import yaml

from models import *
from pathlib import Path
from utils import load_dataset, select_model, progress_bar, train, test
import create_sparse
import dynamic_sparsity


def resume(net, checkpt=None):
    # Load checkpoint.
    global start_epoch
    global best_acc
    if checkpt == None:
        assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
        checkpt = "./checkpoint/ckpt.t7"
    checkpoint = torch.load(checkpt)
    print("Loading checkpoint: ", checkpt)
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    return net, best_acc, start_epoch


def get_schedule_val(schedule, query):
    val = list(schedule[-1].values())[0]
    for i, entry in enumerate(schedule):
        if query < list(entry)[0]:
            val = list(schedule[i - 1].values())[0]
            break
    return val


def adjust_learning_rate(optimizer, epoch, schedule):
    """Sets the learning rate to the initial LR divided by 5 at 30th, 60th and 90th epochs"""
    lr = get_schedule_val(schedule, epoch)

    print("setting learning rate to " + repr(lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def main(args):
    device = f"cuda:{args.gpuid}"
    print("-------------------------------------")
    print("Argument Passed")
    print("Model Selected:", args.model)
    print("Dataset Selected:", args.dataset)
    print("Hyperparameter File:", args.schedule_file)
    print("Message:", args.message)
    print("Num GPUs:", torch.cuda.device_count())
    print("-------------------------------------")

    with open(args.schedule_file, "r") as stream:
        try:
            schedule = yaml.load(stream, yaml.SafeLoader)
        except yaml.YAMLError as exc:
            print(exc)
    print("-------------------------------------")
    print("Hyper Parameters:")
    for k, v in schedule.items():
        print(f"{k}:\t {v}")
    print("-------------------------------------")
    print("-------------------------------------")
    assert args.model, "Error: no model selected!"
    assert args.dataset, "Error: no dataset selected!"
    ngpus_per_node = torch.cuda.device_count()
    assert ngpus_per_node == 1

    best_acc = 0
    start_epoch = 0
    train_acc_list, test_acc_list = [], []
    train_loss_list, test_loss_list = [], []

    tag_string = f"{args.model}_{args.message}_{args.dataset}"
    f"_batch_size_{schedule['train_batch_size']}_{schedule['test_batch_size']}"
    print(f"TagString: {tag_string}")
    print("-------------------------------------")
    print("-------------------------------------")

    # Data
    num_workers = 2
    if args.reproducibility:
        cur_num_workers = 0
        seed_everything()

    (trainloader, testloader, num_channels, num_classes,) = load_dataset(
        args.dataset,
        num_workers,
        schedule["train_batch_size"],
        schedule["test_batch_size"],
    )

    net = select_model(args.model, num_channels, num_classes)
    net = net.to(device)

    if args.resume:
        net = resume(net)

    criterion = nn.CrossEntropyLoss()

    # Training
    num_gpus = torch.cuda.device_count()
    if os.path.exists("./configuration_data_0"):
        os.remove("./configuration_data_0")

    print("Selected Model have {} Convolution Layers: ".format(net.num_conv_layers))
    print("Selected Model have {} Linear Layers: ".format(net.num_linear_layers))
    assert schedule["num_conv_layer"] == net.num_conv_layers

    Path("./dump").mkdir(exist_ok=True, parents=True)
    for i in range(num_gpus):
        for layer in range(schedule["num_conv_layer"]):
            threshold_data = torch.tensor([torch.zeros(1), torch.zeros(1)])
            torch.save(threshold_data, f"./dump/threshold_{layer}_{i}")

    # uncomment for printing the function name
    # import sys
    # sys.settrace(tracefunc)
    if args.inference:
        print("######INFERENCE######")
        assert args.checkpoint != None
        optimizer = optim.SGD(net.parameters(), lr=0.1)
        MODE = yaml_pruning_mode
        LayerSparsity = dynamic_sparsity.layerSparsity(
            optimizer,
            yaml_pruning_type,
            yaml_sparsify_first_layer,
            yaml_sparsify_last_layer,
            inference=True,
        )
        LayerSparsity.add_module(net, 1 - yaml_sparsity, MODE)
        resume(args.checkpoint)
        print("restored_from_epoch", start_epoch)
        print("best_acc", best_acc)
        if yaml_pruning_type == "unstructured":
            create_sparse.print_sparsity_unstructured(net.state_dict())
        elif yaml_pruning_type == "structured_channel":
            create_sparse.print_sparsity_structured(net.state_dict())
        elif yaml_pruning_type == "structured_filter":
            create_sparse.print_sparsity_structured(net.state_dict())
        test(start_epoch + 1, LayerSparsity=LayerSparsity)
        print("######INFERENCE######")

    else:
        # Resume is not allowed for training
        if args.resume:
            assert 0

        # Default LR=0.1
        optimizer = optim.SGD(
            net.parameters(),
            lr=0.1,
            momentum=schedule["momentum"],
            weight_decay=schedule["weight_decay"],
        )
        print("-------------------------------------")
        print("-------------------------------------")
        print("Training Started")
        MODE = schedule["pruning_mode"]
        LayerSparsity = dynamic_sparsity.layerSparsity(
            optimizer,
            schedule["pruning_type"],
            schedule["sparsify_first_layer"],
            schedule["sparsify_last_layer"],
        )
        LayerSparsity.add_module(net, 1 - schedule["sparsity"], MODE)

        for epoch in range(0, schedule["total_epoch"]):
            adjust_learning_rate(optimizer, epoch, schedule["lr_schedule"])
            train(
                net,
                epoch,
                schedule,
                trainloader,
                optimizer,
                device,
                LayerSparsity=LayerSparsity,
            )
            test(net, epoch, args, schedule, testloader, LayerSparsity=LayerSparsity)

        checkpoint = torch.load("./checkpoint/best.t7")
        net.load_state_dict(checkpoint["net"])
        sparsity_budget = LayerSparsity.get_sparsity()
        if yaml_pruning_type == "unstructured":
            net.load_state_dict(
                create_sparse.sparse_unstructured(net.state_dict(), sparsity_budget)
            )
            create_sparse.print_sparsity_unstructured(net.state_dict())
        else:
            if yaml_pruning_type == "structured_channel":
                net.load_state_dict(
                    create_sparse.sparse_channel(net.state_dict(), sparsity_budget)
                )
                create_sparse.print_sparsity_structured(net.state_dict())
            elif yaml_pruning_type == "structured_filter":
                net.load_state_dict(
                    create_sparse.sparse_filter(net.state_dict(), sparsity_budget)
                )
                create_sparse.print_sparsity_structured(net.state_dict())
            else:
                print("ERROR")
        top1_acc = test(schedule["total_epoch"], LayerSparsity=LayerSparsity)
        state = {
            "net": net.state_dict(),
            "acc": top1_acc,
            "epoch": yaml_total_epoch,
            "schedule": schedule,
        }
        torch.save(state, "./checkpoint/pruned_best_" + tag_string + ".t7")
        shutil.copy("./checkpoint/best.t7", "./checkpoint/ckpt_" + tag_string + ".t7")
        acc_dict = {
            "TrainAccuracy": train_acc_list,
            "TestAccuracy": test_acc_list,
            "TrainLoss": train_loss_list,
            "TestLoss": test_loss_list,
        }
        np.save("./checkpoint/acc_log_" + tag_string + ".npy", acc_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("--gpuid", "-g", type=int, default=0, help="gpuid ")
    parser.add_argument(
        "--resume", "-r", action="store_true", help="resume from checkpoint"
    )
    parser.add_argument("--model", "-model", type=str, help="model")
    parser.add_argument("--dataset", "-dataset", type=str, help="dataset")
    parser.add_argument(
        "--message",
        "-m",
        type=str,
        default="",
        help="appended in the name of checkpoint and log file",
    )
    parser.add_argument(
        "--inference", "-inference", default=0, type=int, help="inference"
    )
    parser.add_argument(
        "--checkpoint", "-checkpoint", default=None, type=str, help="ckpt"
    )
    parser.add_argument(
        "--progress_bar", "-p", default=0, type=int, help="progress bar"
    )
    parser.add_argument(
        "--dump_parameter",
        "-dump_parameter",
        default=0,
        type=int,
        help="dump_parameter",
    )
    parser.add_argument(
        "--reproducibility",
        "-reproducibility",
        default=0,
        type=int,
        help="reproducibility",
    )
    parser.add_argument(
        "--schedule-file",
        default="./schedule.yaml",
        type=str,
        help="yaml file containing learning rate schedule and rewire period schedule",
    )

    args = parser.parse_args()
    main(args)
