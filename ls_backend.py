from asyncore import read
import os
import json
from tkinter import image_names
from PIL import Image
from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np
from pipeline import PIPELINE
import codecs
import random
reader = PIPELINE()


def create_image_url(filepath):
    """
    Label Studio requires image URLs, so this defines the mapping from filesystem to URLs
    if you use ./serve_local_files.sh <my-images-dir>, the image URLs are localhost:8081/filename.png
    Otherwise you can build links like /data/upload/filename.png to refer to the files
    """
    filename = os.path.basename(filepath)
    return f'http://localhost:8080/{filename}'


def convert_to_ls(image, tesseract_output, per_level='block_num'):
    """
    :param image: PIL image object
    :param tesseract_output: the output from tesseract
    :param per_level: control the granularity of bboxes from tesseract
    :return: tasks.json ready to be imported into Label Studio with "Optical Character Recognition" template
    """
    image_width, image_height = image.size
    results = []
    all_scores = []
    for i, level_idx in enumerate(tesseract_output['level']):
        if level_idx == per_level_idx:
            bbox = {
                'x': 100 * tesseract_output['left'][i] / image_width,
                'y': 100 * tesseract_output['top'][i] / image_height,
                'width': 100 * tesseract_output['width'][i] / image_width,
                'height': 100 * tesseract_output['height'][i] / image_height,
                'rotation': 0
            }

            words, confidences = [], []
            for j, curr_id in enumerate(tesseract_output[per_level]):
                if curr_id != tesseract_output[per_level][i]:
                    continue
                word = tesseract_output['text'][j]
                confidence = tesseract_output['conf'][j]
                words.append(word)
                if confidence != '-1':
                    confidences.append(float(float(confidence )/ 100.))

            text = ' '.join(words).strip()
            if not text:
                continue
            region_id = str(uuid4())[:10]
            score = sum(confidences) / len(confidences) if confidences else 0
            bbox_result = {
                'id': region_id, 'from_name': 'bbox', 'to_name': 'image', 'type': 'rectangle',
                'value': bbox}
            transcription_result = {
                'id': region_id, 'from_name': 'transcription', 'to_name': 'image', 'type': 'textarea',
                'value': dict(text=[text], **bbox), 'score': score}
            results.extend([bbox_result, transcription_result])
            all_scores.append(score)

    return {
        'data': {
            'ocr': create_image_url(image.filename)
        },
        'predictions': [{
            'result': results,
            'score': sum(all_scores) / len(all_scores) if all_scores else 0
        }]
    }

def convert_ls(image, image_name, bbox, texts):
    
    h = image.shape[0]
    w = image.shape[1]

    results = []
    all_scores = []
    confidences = []

    for item, words in zip(bbox, texts):

        bbox = {
                'x': 100 * item[0][0] / w,
                'y': 100* item[0][1] / h,
                'width': 100 * (item[1][0] - item[0][0]) / w,
                'height': 100 * (item[3][1] - item[0][1]) / h,
                'rotation': 0
            }
        
        score = random.uniform(0.5, 1)

        region_id = str(uuid4())[:10]
        score = sum(confidences) / len(confidences) if confidences else 0
        bbox_result = {
            'id': region_id, 'from_name': 'bbox', 'to_name': 'image', 'type': 'rectangle',
            'value': bbox}
        transcription_result = {
            'id': region_id, 'from_name': 'transcription', 'to_name': 'image', 'type': 'textarea',
            'value': dict(text=[words], **bbox), 'score': score}
        results.extend([bbox_result, transcription_result])
        all_scores.append(score)
        

    return {
        'data': {
            'ocr': create_image_url(image_name)
        },
        'predictions': [{
            'result': results,
            'score': sum(all_scores) / len(all_scores) if all_scores else 0
        }]
    }


tasks = []
# collect the receipt images from the image directory

img_list = os.listdir('images')
img_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

for item in img_list: 
    img = cv2.imread(f'images/{item}')
    ratio = img.shape[0]/512
    pd_results, texts = reader.infer(img)

    bbox = np.array(pd_results[1])
    bbox = bbox * ratio

    task = convert_ls(img, item, bbox, texts)
    tasks.append(task)

# create a file to import into Label Studio
with codecs.open('ocr_tasks.json', 'w', encoding='utf-8') as f:
    json.dump(tasks, f, indent=2, ensure_ascii=False)


