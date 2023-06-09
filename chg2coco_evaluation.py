import json
import os
from collections import defaultdict

# Initialize the category IDs
category_ids = {'BIC': 1}

# Initialize the categories
categories = [{'id': id, 'name': name} for name, id in category_ids.items()]

def chg2coco(json_dir):
    # Create a dictionary to store the COCO format dataset
    coco_dataset = defaultdict(list)
    coco_dataset['categories'] = categories
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(json_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                # Load the JSON file
                data = json.load(f)
                # Extract the image information
                image_info = {
                    'id': len(coco_dataset['images']) + 1,
                    'file_name': data['images']['filename'],
                    'height': data['images']['height'],
                    'width': data['images']['width'],
                }
                coco_dataset['images'].append(image_info)
                
                # Loop through all the annotations and extract the bounding box information
                for annotation in data['annotations']:
                    bbox = annotation['bbox']
                    if bbox['classid'] == 'BIC': # check if the classid is 'BIC'
                        points = bbox['points']
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        x3, y3 = points[2]
                        x4, y4 = points[3]
                        x = min(x1, x2, x3, x4)
                        y = min(y1, y2, y3, y4)
                        w = max(x1, x2, x3, x4) - x
                        h = max(y1, y2, y3, y4) - y
                        bbox_info = {
                            'image_id': image_info['id'],
                            'category_id': category_ids[bbox['classid']],
                            'bbox': [x, y, w, h],
                            'id': len(coco_dataset['annotations']) + 1,
                            'iscrowd': 0, # set to 0 for object detection
                            'area': w * h # calculate area
                        }

                        # Add additional information to the bounding box if available
                        if 'text' in bbox:
                            bbox_info['text'] = bbox['text']
                        if 'ocrdirection' in bbox:
                            bbox_info['ocrdirection'] = bbox['ocrdirection']

                        coco_dataset['annotations'].append(bbox_info)
    return coco_dataset


eval_dir = 'evaluation/json'

with open(os.path.join('evaluation', 'coco_dataset_BIC.json'), 'w') as f:
    json.dump(chg2coco(eval_dir), f)

    print('evaluation', "경로에 coco형식의 json파일을 생성했습니다.")