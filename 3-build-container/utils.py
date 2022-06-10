import os
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset

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

def save_checkpoint(checkpoint_file, epoch, model, optimizer):
    checkpoint_dir = os.path.dirname(checkpoint_file)
    os.makedirs(checkpoint_dir, exist_ok=True)

    snapshot = {
        "epoch": epoch,
        "state_dict":  model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    torch.save(snapshot, checkpoint_file)
    print(f"=> saved checkpoint for epoch {epoch+1} at {checkpoint_file}")

def load_checkpoint(checkpoint_file, model, optimizer):
    if os.path.isfile(checkpoint_file):
        print('loading checkpoint file:', checkpoint_file)
        snapshot = torch.load(checkpoint_file)
        epoch = snapshot["epoch"] + 1 # start from next epoch
        model.load_state_dict(snapshot["state_dict"])
        optimizer.load_state_dict(snapshot["optimizer"])
        print("Restored model from previous checkpoint")
    else:
        epoch = 0

    return epoch
