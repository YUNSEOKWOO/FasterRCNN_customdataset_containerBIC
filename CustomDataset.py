import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import json
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
import numpy as np
import os
from PIL import Image



class MyDataset(Dataset):
    def __init__(self, image_dir, json_file, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir

        # Load the JSON file
        with open(json_file) as f:
            self.data = json.load(f)

        self.images = self.data["images"]
        self.annotations = self.data["annotations"]

        # Remove duplicate image_ids from annotations
        unique_image_ids = set()
        self.unique_annotations = []
        for ann in self.annotations:
            if ann["image_id"] not in unique_image_ids:
                unique_image_ids.add(ann["image_id"])
                self.unique_annotations.append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx]["file_name"])
        image = Image.open(image_path).convert("RGB")

        # Get the annotations for this image
        boxes = []
        labels = []
        areas = []
        iscrowds = []
        image_id = []
        for ann in self.unique_annotations:
            if ann["image_id"] == idx + 1:
                bbox = ann["bbox"]
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # Convert from [x, y, w, h] to [x1, y1, x2, y2]
                boxes.append(bbox)
                labels.append(ann["category_id"])
                areas.append(bbox[2] * bbox[3])
                iscrowds.append(int(0))
                image_id.append(ann["image_id"])

        # Apply transforms if provided
        if self.transforms:
            image = self.transforms(image)

        # Convert annotations to PyTorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowds = torch.as_tensor(iscrowds, dtype=torch.uint8)
        image_id = torch.as_tensor(image_id, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowds
        }

        return image, target