import random
import time
import os
import argparse

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms

import matplotlib.pyplot as plt

from events import EventWriter

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, choices=["dataset3", "dataset4", "mixed"])
parser.add_argument("--ft", type=str, choices=["classifier", "model"])
parser.add_argument("--load_model_from")
args = parser.parse_args()

dataset_name = args.dataset_name
event_path = "./events"
epochs = 200
lr = 0.001
momentum = 0.9
split = 0.7
model_name = "mobilenet_v3_small"
finetune_classifier_only = args.ft == "classifier"

finetune_key = "finetune_classifier_only" if finetune_classifier_only else "finetune_full_model"
run_key = f"{model_name}-{dataset_name}-{finetune_key}"

best_model_params_path = f"./{run_key}-best_model_params.pt"


print(run_key)


NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
INPUT_IMAGE_SIZE = 224

CPU_ONLY = False
STOP_AT_100 = False
SAME_VAL_TRANSFORM = False
BATCH_SIZE = 4
N_WORKERS = 16
DROPOUT = 0.2
PREFETCH_FACTOR = 4
PIN_MEMORY = True
NON_BLOCKING = True

TOP_K = 3


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(INPUT_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])

train_dataset = None
val_dataset = None
classes = None
if dataset_name != "mixed":
    dataset_path = f"../data/{dataset_name}"
    train_dataset = ImageFolder(dataset_path, train_transform)
    val_dataset = ImageFolder(dataset_path, val_transform)
    classes = train_dataset.classes
else:
    dataset_path = f"../data/dataset3"
    train_dataset3 = ImageFolder(dataset_path, train_transform)
    val_dataset3 = ImageFolder(dataset_path, val_transform)

    dataset_path = f"../data/dataset4"
    train_dataset4 = ImageFolder(dataset_path, train_transform)
    val_dataset4 = ImageFolder(dataset_path, val_transform)

    train_dataset = torch.utils.data.ConcatDataset([train_dataset3, train_dataset4])
    val_dataset = torch.utils.data.ConcatDataset([val_dataset3, val_dataset4])

    classes = train_dataset3.classes
n_classes = len(classes)

dataset_size = len(train_dataset)
train_size = int(dataset_size * split)
val_size = dataset_size - train_size 

print(f"n_classes={n_classes}")
print(f"split={split}")
print(f"dataset_size={dataset_size}")
print(f"train_size={train_size}")
print(f"val_size={val_size}")
print()

if SAME_VAL_TRANSFORM:
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.7, 0.3])
else:
    indices = list(range(dataset_size))
    random.shuffle(indices)

    train_indicies = indices[:train_size]
    val_indicies = indices[train_size:]

    train_dataset = torch.utils.data.Subset(train_dataset, train_indicies)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indicies)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=N_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    pin_memory=PIN_MEMORY,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=N_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    pin_memory=PIN_MEMORY,
)

def freeze_all_params(model):
    for param in model.parameters():
        param.requires_grad = False

if model_name == "resnet18":
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    if finetune_classifier_only:
        freeze_all_params(model)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, n_classes)
elif model_name == "mobilenet_v2":
    model = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
    if finetune_classifier_only:
        freeze_all_params(model)
    num_features = model.last_channel
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT, inplace=True),
        nn.Linear(num_features, n_classes)
    )
elif model_name == "mobilenet_v3_small":
    model = torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    if finetune_classifier_only:
        freeze_all_params(model)
    inverted_residual_setting, last_channel = torchvision.models.mobilenetv3._mobilenet_v3_conf("mobilenet_v3_small")
    lastconv_input_channels = inverted_residual_setting[-1].out_channels
    lastconv_output_channels = 6 * lastconv_input_channels
    model.classifier = nn.Sequential(
        nn.Linear(lastconv_output_channels, last_channel),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=DROPOUT, inplace=True),
        nn.Linear(last_channel, n_classes),
    )
elif model_name == "mobilenet_v3_large":
    model = torchvision.models.mobilenet_v3_large(weights='IMAGENET1K_V2')
    if finetune_classifier_only:
        freeze_all_params(model)
    inverted_residual_setting, last_channel = torchvision.models.mobilenetv3._mobilenet_v3_conf("mobilenet_v3_large")
    lastconv_input_channels = inverted_residual_setting[-1].out_channels
    lastconv_output_channels = 6 * lastconv_input_channels
    model.classifier = nn.Sequential(
        nn.Linear(lastconv_output_channels, last_channel),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=DROPOUT, inplace=True),
        nn.Linear(last_channel, n_classes),
    )
else:
    raise ValueError(f"Invalid model_name: {model_name}")

device = None
if CPU_ONLY:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.load_model_from:
    print(f"Loading model from path={args.load_model_from}")
    # TODO: probably need to freeze if we only want to fine tune the classifier?
    model.load_state_dict(torch.load(args.load_model_from, weights_only=True))

print(f"using device={device}")
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# optimizer = optim.RMSprop(model.parameters(), momentum=momentum)
# optimizer = optim.Adam(model.parameters(), lr=lr)

# TODO: Top 5.
event_writer = EventWriter(event_path)
best_train_output = []
best_validation_output = []

since = time.time()
torch.save(model.state_dict(), best_model_params_path)
best_train_accuracy = 0.0
best_accuracy = 0.0

for epoch in range(epochs):
    print(f'epoch={epoch}')
    epoch_start_time = time.time()

    # Training
    model.train()

    loss = 0.0
    correct = 0
    last_train_output = []

    for inputs, labels in train_dataloader:
        inputs = inputs.to(device, non_blocking=NON_BLOCKING)
        labels = labels.to(device, non_blocking=NON_BLOCKING)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)

            for (p, l) in zip(preds, labels.data):
                last_train_output.append((p.item(), l.item()))

            loss.backward()
            optimizer.step()

        loss += loss.item() * inputs.size(0)
        correct += torch.sum(preds == labels.data)

    train_epoch_loss = loss / float(train_size)
    train_epoch_accuracy = 100. * correct.double() / float(train_size)

    if train_epoch_accuracy > best_train_accuracy:
        best_train_accuracy = train_epoch_accuracy
        best_train_output = last_train_output

    print(f'training: loss={train_epoch_loss:.4f}, accuracy={train_epoch_accuracy:.4f}% ({correct}/{train_size})')

    # Validation
    model.eval()

    loss = 0.0
    correct = 0
    top_k = 0

    last_validation_output = []
    for inputs, labels in val_dataloader:
        inputs = inputs.to(device, non_blocking=NON_BLOCKING)
        labels = labels.to(device, non_blocking=NON_BLOCKING)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)

            _, top_k_preds = torch.topk(outputs, TOP_K)
            top_k += sum([want.item() in set(got.tolist()) for (want, got) in zip(labels, top_k_preds)])

        loss += loss.item() * inputs.size(0)
        correct += torch.sum(preds == labels.data)

        for (p, l) in zip(preds, labels.data):
            last_validation_output.append((p.item(), l.item()))

    val_epoch_loss = loss / float(val_size)
    val_epoch_accuracy = 100. * correct.double() / float(val_size)
    top_k = top_k / float(val_size)

    event_writer.add(f"{run_key}-per_epoch", {
        "epoch": epoch,
        "train_epoch_loss": train_epoch_loss.item(),
        "train_epoch_accuracy": train_epoch_accuracy.item(),
        "val_epoch_loss": val_epoch_loss.item(),
        "val_epoch_accuracy": val_epoch_accuracy.item(),
        "top_k": top_k,
    })
    print(f'validation: loss={val_epoch_loss:.4f}, accuracy={val_epoch_accuracy:.4f}% ({correct}/{val_size}), best_accuracy={best_accuracy:.4f} top_{TOP_K}={top_k:.4f}')
    epoch_time_elapsed = time.time() - epoch_start_time
    print(f'epoch took={epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s')

    if val_epoch_accuracy > best_accuracy:
        best_accuracy = val_epoch_accuracy
        best_validation_output = last_validation_output
        torch.save(model.state_dict(), best_model_params_path)

    if STOP_AT_100 and correct == val_size:
        print(f"100% validation accuracy: stopping early.")
        break


    print()

time_elapsed = time.time() - since
print(f'took={time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Best validation accuracy={best_accuracy:4f}')

prediction_key = f"{run_key}-train_predictions"
for (p, l) in best_train_output:
    event_writer.add(prediction_key, {
        "pred": p,
        "label": l,
    })

prediction_key = f"{run_key}-val_predictions"
for (p, l) in best_validation_output:
    event_writer.add(prediction_key, {
        "pred": p,
        "label": l,
    })
