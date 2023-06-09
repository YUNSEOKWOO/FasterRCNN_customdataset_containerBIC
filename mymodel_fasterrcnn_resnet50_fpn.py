import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator

def mymodel(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Get the current anchor generator
    # anchor_generator = model.rpn.anchor_generator
    # # Create a new anchor generator with adjusted aspect ratios
    # new_anchor_generator = AnchorGenerator(sizes=anchor_generator.sizes, aspect_ratios=((3.0, 5.0, 7.0)))
    # # Update the anchor generator in the model
    # model.rpn.anchor_generator = new_anchor_generator

    return model