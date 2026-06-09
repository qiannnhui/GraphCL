import torch
from torch_geometric.data import DataLoader
from aug import TUDataset_aug
from torch_geometric.datasets import TUDataset
import sys
import psutil
import os

def print_mem(msg):
    process = psutil.Process(os.getpid())
    print(f"[{msg}] Memory Use: {process.memory_info().rss / 1024**2:.2f} MB")

print_mem("Start")
path = '/home/qiannnhui/GraphCL/unsupervised_TU/data'
ds_name = sys.argv[1]

print(f"Loading {ds_name} Dataset Eval...")
dataset_eval = TUDataset(path, name=ds_name)
print_mem("Loaded eval dataset")

print(f"Loading {ds_name} Dataset Aug...")
dataset = TUDataset_aug(path, name=ds_name, aug='dnodes', aug_ratio=0.1)
print_mem("Loaded aug dataset")

print("Creating dataloaders...")
dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
dataloader_eval = DataLoader(dataset_eval, batch_size=32, num_workers=4, pin_memory=True)
print_mem("Created dataloaders")

print("Fetching batch_all...")
tmp_loader = DataLoader(dataset_eval, batch_size=len(dataset_eval), shuffle=False)
batch_all = next(iter(tmp_loader))
print_mem("Fetched batch_all")

print("Fetching first batch from dataloader...")
batch = next(iter(dataloader))
print_mem("Fetched first batch")

print("SUCCESS")
