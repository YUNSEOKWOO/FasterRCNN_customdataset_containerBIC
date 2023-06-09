import subprocess
import sys
subprocess.run(['git', 'clone', 'https://github.com/pytorch/vision.git'])
sys.path.append('vision/references/detection')
sys.path.append('cocoapi/PythonAPI')
from pycocotools import *
from engine import train_one_epoch, evaluate
import utils

import torch
import os
from CustomDataset import MyDataset
from get_transform import get_transform
from mymodel_fasterrcnn_mobilenet_v3_large_fpn import mymodel
from engine import train_one_epoch, evaluate
import utils

# Set paths for training and evaluation
train_image_dir = "train/image"
train_json_file = "train/coco_dataset_BIC.json"
eval_image_dir = "evaluation/image"
eval_json_file = "evaluation/coco_dataset_BIC.json"

# Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the number of classes
num_classes = 2

# Create dataset and apply transformations
dataset = MyDataset(image_dir=train_image_dir, json_file=train_json_file, transforms=get_transform())
dataset_test = MyDataset(image_dir=eval_image_dir, json_file=eval_json_file, transforms=get_transform())

# Define data loaders for training and validation
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=16, shuffle=True, num_workers=1, collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=16, shuffle=False, num_workers=1, collate_fn=utils.collate_fn)

# Initialize the model
model = mymodel(num_classes)

# Move the model to the device
model.to(device)

# Set up optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Create directory for saving model weights
model_save_dir = "model"
os.makedirs(model_save_dir, exist_ok=True)

# Define the number of epochs
num_epochs = 10

def main():
    for epoch in range(num_epochs):
        # Train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=150)

        # Update the learning rate
        lr_scheduler.step()

        # Save model weights
        model_save_path = os.path.join(model_save_dir, f"model_weights_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)

        # Evaluate the model
        evaluator = evaluate(model, data_loader_test, device=device)

        # Get evaluation metrics
        coco_eval = evaluator.coco_eval["bbox"]
        stats = coco_eval.stats

        # Print evaluation metrics
        print(f"Epoch: {epoch+1}")
        print(f"mAP: {stats[0]}")
        print(f"Precision: {stats[1]}")
        print(f"Recall: {stats[2]}")
        print()

if __name__ == '__main__':
    main()
