"""Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init

from torchvision import transforms
from torchvision import datasets
from models import *


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("==> Computing mean and std..")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode="fan_out")
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
term_width = 80

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append("  Step: %s" % format_time(step_time))
    L.append(" | Tot: %s" % format_time(tot_time))
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def load_dataset(dataset_name, num_workers, train_batch_size, test_batch_size):
    print("==> Preparing data..")
    print(f"Dataset Selected {dataset_name}")
    if dataset_name not in ["Mnist", "Cifar10", "Cifar100"]:
        print("Dataset not found")
        assert 0

    # Assume CIFAR
    mean_std_cifar = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        + mean_std_cifar
    )
    transform_test = transforms.Compose(mean_std_cifar)

    if dataset_name == "Cifar10":
        num_classes = 10
        num_channels = 3
        dataloader = datasets.CIFAR10
        datadirectory = "./data/cifar10"

    elif dataset_name == "Cifar100":
        num_classes = 100
        num_channels = 3
        dataloader = datasets.CIFAR100
        datadirectory = "./data/cifar100"

    elif dataset_name == "Mnist":
        num_classes = 10
        num_channels = 1
        dataloader = datasets.MNIST
        datadirectory = "./data/mnist"
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,)
                ),  # mean and std of mnist dataset
            ]
        )
        transform_test = transform_train

    ## Create Dataloaders
    trainset = dataloader(
        root=datadirectory, train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, train_batch_size, shuffle=True, num_workers=num_workers
    )
    testset = dataloader(
        root=datadirectory, train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, test_batch_size, shuffle=False, num_workers=num_workers
    )

    return trainloader, testloader, num_channels, num_classes


def select_model(x, numchannels, numclasses):
    return {
        "ResNet18": ResNet18(numchannels, numclasses),
        "ResNet34": ResNet34(numchannels, numclasses),
        "ResNet50": ResNet50(numchannels, numclasses),
        "ResNet101": ResNet101(numchannels, numclasses),
        "WRN-28-10": Wide_ResNet(28, 10, 0.3, numclasses),
        "WRN-16-8": Wide_ResNet(16, 8, 0.3, numclasses),
        "VGG16": VGG("VGG16", numclasses),
        "DenseNet121": DenseNet121(numclasses),
    }[x]


def train(net, epoch, schedule, trainloader, optimizer, device, LayerSparsity=None):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    sparsity_budget = LayerSparsity.get_sparsity()
    configuration_data = {
        "epoch": epoch,
        "batch_idx": 0,
        "layer": 0,
        "type": 0,
        "period": schedule["topk_period"],
        "sparsity": sparsity_budget,
        "warmup": schedule["warmup"],
        "global_sparsity": schedule["sparsity"],
        "pruning_type": schedule["pruning_type"],
    }
    torch.save(configuration_data, "configuration_data_0")
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        sparsity_budget = LayerSparsity.get_sparsity()
        configuration_data = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "layer": 0,
            "type": 0,
            "period": schedule["topk_period"],
            "sparsity": sparsity_budget,
            "warmup": schedule["warmup"],
            "global_sparsity": schedule["sparsity"],
            "pruning_type": schedule["pruning_type"],
        }
        torch.save(configuration_data, "configuration_data_0")
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()

        LayerSparsity.gather_statistic()
        optimizer.step()
        LayerSparsity.step()

        sparsity_budget = LayerSparsity.get_sparsity()
        torch.save(sparsity_budget, "sparsity_configuration_0")

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if args.progress_bar:
            progress_bar(
                batch_idx,
                len(trainloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    train_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    acc = 100.0 * correct / total
    train_acc_list.append(acc)
    cum_train_loss = train_loss / (batch_idx + 1)
    train_loss_list.append(cum_train_loss)
    print("Epoch: ", epoch, "Train Accuracy: ", acc)


def test(epoch, LayerSparsity=None):
    global best_acc
    global test_acc_list
    global test_loss_list
    global loaded_schedule
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    sparsity_budget = LayerSparsity.get_sparsity()
    configuration_data = {
        "epoch": 1000,
        "batch_idx": 0,
        "layer": 0,
        "type": 1,
        "period": yaml_topk_period,
        "sparsity": sparsity_budget,
        "warmup": yaml_warmup,
        "global_sparsity": yaml_sparsity,
        "pruning_type": yaml_pruning_type,
    }
    torch.save(configuration_data, "configuration_data_0")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs, targets = inputs.to(device), targets.to(device)

            sparsity_budget = LayerSparsity.get_sparsity()
            configuration_data = {
                "epoch": epoch,
                "batch_idx": batch_idx,
                "layer": 0,
                "type": 1,
                "period": yaml_topk_period,
                "sparsity": sparsity_budget,
                "warmup": yaml_warmup,
                "global_sparsity": yaml_sparsity,
                "pruning_type": yaml_pruning_type,
            }
            torch.save(configuration_data, "configuration_data_0")
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if args.progress_bar:
                progress_bar(
                    batch_idx,
                    len(testloader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        test_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )

    # Save checkpoint.
    acc = 100.0 * correct / total
    test_acc_list.append(acc)
    cum_test_loss = (test_loss) / (batch_idx + 1)
    test_loss_list.append(cum_test_loss)
    state = {
        "net": net.state_dict(),
        "acc": acc,
        "epoch": epoch,
        "loaded_schedule": loaded_schedule,
    }
    print("Epoch: ", epoch, "Test Accuracy: ", acc)

    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")

    if (
        1
    ):  # acc > best_acc: #comment the line for not using greedy algorithm in that scenario best_acc will become the acc
        print("Saving..")
        torch.save(state, "./checkpoint/ckpt.t7")
        if (
            acc > best_acc
        ):  # comment the line for not using greedy algorithm in that scenario best_acc will become the acc
            torch.save(state, "./checkpoint/best.t7")
            best_acc = acc

    if args.dump_parameter:
        torch.save(
            state, "./checkpoint/ckpt_" + tag_string + "_Epoch" + str(epoch) + ".t7"
        )

    return acc
