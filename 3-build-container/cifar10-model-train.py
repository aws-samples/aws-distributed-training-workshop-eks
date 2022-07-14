import os
from datetime import timedelta
import argparse

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from torch.distributed import init_process_group
from torch.utils.tensorboard import SummaryWriter

from cnn_model import MyCnnModel # custom cnn model
from utils import *

parser = argparse.ArgumentParser(description="PyTorch Elastic cifar10 Training")
parser.add_argument("data", help="path to dataset")
parser.add_argument('--workers', default=32, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run (default: 10)')
parser.add_argument('--batch-size', default=256, type=int,
                    help='mini-batch size on each node (default: 256)')
parser.add_argument('--learning-rate', default=0.001, type=float,
                    help='learning rate (default: 0.001')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=5, type=int,
                    help='print frequency (default: 5)')
parser.add_argument('--model-file', default='/efs-shared/cifar10_model.pth', type=str,
                    help='filename with path to save model (default: /efs-shared/cifar10_model.pth')
parser.add_argument("--checkpoint-file", default="/efs-shared/checkpoint.pth.tar", type=str,
                    help="checkpoint file path, to load and save to")


def cifar10_train_dataloader(data_dir, batch_size, num_data_workers):
    files = ['data_batch_'+str(i+1) for i in range(5)]

    train_images = []
    train_labels = []
    for file in files:
        images, labels = unpickle(data_dir + file)
        train_images.extend(images)
        train_labels.extend(labels)

    # convert numpy arrays to torch TensorDataset
    train_dataset = get_tensordataset(train_images, train_labels)

    train_sampler = ElasticDistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_data_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    return train_loader


def initialize_model(lr, momentum, weight_decay):
    model = MyCnnModel()
    model = DistributedDataParallel(model)

    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

    return model, criterion, optimizer


def main():
    print("Main function called ...")
    #import debugpy; debugpy.listen(('0.0.0.0',5678)); debugpy.wait_for_client(); breakpoint()
    init_process_group(backend="gloo", init_method="env://", timeout=timedelta(seconds=10))
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    modelFile = args.model_file
    tensorDir = "%s/runs/"%(os.path.dirname(modelFile))
    writer = SummaryWriter(log_dir=tensorDir)

    print("reading", args.data)
    train_loader = cifar10_train_dataloader(args.data, args.batch_size, args.workers)
    model, criterion, optimizer = initialize_model(args.learning_rate, args.momentum, args.weight_decay)

    processor = os.getenv("PROCESSOR","cpu")
    print("Desired processor type: %s"%processor)

    device_type="cpu"
    device=torch.device("cpu")

    if processor == "gpu":
        if torch.cuda.is_available():
            device=torch.device("cuda")
            device_type="gpu"
            model.to(device)
        else:
            print("torch.cuda.is_available() returned False!")

    print("Running on processor type: %s"%(device_type))
    
    start_epoch = 0
    # load previously stored checkpoint if it exists.
    start_epoch = load_checkpoint(args.checkpoint_file, model, optimizer)

    model.train()
    for epoch in range(start_epoch, args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if device_type == "gpu":
              inputs = inputs.to(device)
              labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % args.print_freq == args.print_freq-1:    # print every args.print_freq mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / args.print_freq:.3f}')
                writer.add_scalar('Epoch', epoch+1)
                writer.add_scalar('Iteration', i+1)
                writer.add_scalar('Loss', running_loss / args.print_freq)
                running_loss = 0.0
            
        if rank==0: # Only one pod will save the checkpoint
            save_checkpoint(args.checkpoint_file, epoch, model, optimizer)

    if rank==0:  # Only one pod will save the final model
        print('saving final model:', args.model_file)
        torch.save(model.module.state_dict(), args.model_file)
    
    writer.close()

if __name__ == "__main__":
    main()
    print('Finished Training')
