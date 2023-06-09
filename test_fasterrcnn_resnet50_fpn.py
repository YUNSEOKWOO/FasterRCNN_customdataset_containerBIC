import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import shutil

# Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved model
num_classes = 2
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("model/fasterrcnn_resnet50_fpn/model_weights_epoch_10.pth"))
model.to(device)
model.eval()

# Function to perform prediction on an image
def predict_image(image_path, model, score_threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    image_tensor = torchvision.transforms.functional.to_tensor(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image_tensor)
    return filter_prediction(prediction[0], score_threshold)

# Function to filter the predicted bounding boxes based on score threshold
def filter_prediction(prediction, score_threshold):
    boxes = prediction['boxes'].detach().cpu().numpy()
    labels = prediction['labels'].detach().cpu().numpy()
    scores = prediction['scores'].detach().cpu().numpy()
    
    # Filter bounding boxes based on score threshold
    high_score_indices = scores >= score_threshold
    filtered_boxes = boxes[high_score_indices]
    filtered_labels = labels[high_score_indices]
    filtered_scores = scores[high_score_indices]
    
    return filtered_boxes, filtered_labels, filtered_scores

# Set the paths for the directories
image_dir = r"real"
result_dir = r"real/fasterrcnn_resnet50_fpn"

# Set the score threshold for filtering bounding boxes
score_threshold = 0.5

# Get the list of image files in the directory
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith((".jpg", ".png"))]

# Loop over the image files and perform predictions
for image_file in image_files:
    # Perform prediction and filter bounding boxes
    boxes, labels, scores = predict_image(image_file, model, score_threshold)

    # Skip processing if no detections were made
    if len(boxes) == 0:
        print(f"No detections found for {image_file}. Saving the original image...")
        
        # Save the original image as the result image
        file_name = os.path.basename(image_file)
        save_path = os.path.join(result_dir, "result_" + file_name)
        shutil.copy(image_file, save_path)
        
        print(f"Saved original image: {save_path}")
        continue

    # Load the image using PIL
    image = Image.open(image_file)

    # Create a figure and axes with the same size as the original image
    fig, ax = plt.subplots(1, figsize=(image.width/100, image.height/100), dpi=100)

    # Remove axis
    ax.axis('off')

    # Display the image
    ax.imshow(image)

    # Find the index of the bounding box with the highest score
    max_score_index = scores.argmax()

    # Get the bounding box, label, and score with the highest score
    max_box = boxes[max_score_index]
    max_label = labels[max_score_index]
    max_score = scores[max_score_index]

    # Replace label "1" with "BIC"
    if max_label == 1:
        max_label = "BIC"

    # Add the rectangle patch for the bounding box with the highest score
    rect = patches.Rectangle((max_box[0], max_box[1]), max_box[2] - max_box[0], max_box[3] - max_box[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Add label and score for the bounding box with the highest score
    label_text = f"{max_label}: {max_score:.2f}"
    ax.text(max_box[0], max_box[1], label_text, color='r')

    # Save the plot with the same name as the original image in the result directory
    file_name = os.path.basename(image_file)
    save_path = os.path.join(result_dir, "result_" + file_name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Saved predicted image: {save_path}")
