import os
import pickle
import numpy as np
from datetime import timedelta
import argparse

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, models
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group

from cnn_model import MyCnnModel # custom cnn model

parser = argparse.ArgumentParser(description="PyTorch Elastic cifar10 Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument('--workers', default=32, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('--batch-size', default=128, type=int,
                    help='mini-batch size on each node (default: 128)')
parser.add_argument('--model-file', default='/efs-shared/cifar10_model.pth', type=str,
                    help='filename with path to save model (default: /efs-shared/cifar10_model.pth')


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data[b'data'], data[b'labels']


def get_tensordataset(images, labels):
    images_arr = np.array(images)
    images_arr = np.reshape(images_arr, (-1,3,32,32))
    labels_arr = np.array(labels)
    images_arr = images_arr/255.  # normalize

    tensor_X = torch.tensor(images_arr, dtype=torch.float32)
    tensor_y = torch.tensor(labels_arr, dtype=torch.long)
    dataset = TensorDataset(tensor_X, tensor_y)

    return dataset


def cifar10_test_dataloader(data_dir, batch_size, num_data_workers):
    test_images, test_labels = unpickle(data_dir + 'test_batch')
    test_dataset = get_tensordataset(test_images, test_labels)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_data_workers,
        pin_memory=True,
    )

    return test_loader


def main():
    args = parser.parse_args()
    print("reading", args.data)
    test_loader = cifar10_test_dataloader(args.data, args.batch_size, args.workers)
    print('loading model', args.model_file)
    model = MyCnnModel()
    model.load_state_dict(torch.load(args.model_file))

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct = 0
    total = 0

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)        

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

            
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if __name__ == "__main__":
    main()
    print('Finished Testing')
