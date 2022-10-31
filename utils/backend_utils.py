import copy
import time
import json
import itertools
from functools import wraps
import os

import cv2
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt



def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        delta_time = round(float(end - start), 8)
        #print("- Function {} run in {}'s".format(func.__name__, delta_time))
        return result

    return wrapper


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def sort_box(boxes):
    sorted_boxes = []
    for box in boxes:
        sorted_boxes.append(order_points_clockwise(box))
    mid_points = []
    for box in sorted_boxes:
        try:
            mid = line_intersection((box[0],box[2]), (box[1], box[3]))
            mid_points.append(mid)
        except:
            continue
    sorted_indices = np.argsort(mid_points, axis=0)
    sorted_boxes = sorted(sorted_boxes , key=lambda sorted_indices: [sorted_indices[0][1], sorted_indices[0][0]]) 
    return sorted_boxes

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
    
def crop_box(img, boxes, sort=True):

    img_list = []
    h,w,c = img.shape
    
    if sort:
        boxes = sort_box(boxes)
    
    for i, box in enumerate(boxes):
        
        (x1,y1),(x2,y2),(x3,y3),(x4,y4) = box
        x1,y1,x2,y2,x3,y3,x4,y4 = int(x1),int(y1),int(x2),int(y2),int(x3),int(y3),int(x4),int(y4)
        x1 = max(0, x1)
        x2 = max(0, x2)
        x3 = max(0, x3)
        x4 = max(0, x4)
        y1 = max(0, y1)
        y2 = max(0, y2)
        y3 = max(0, y3)
        y4 = max(0, y4)
        min_x = max(0, min(x1,x2,x3,x4))
        min_y = max(0, min(y1,y2,y3,y4))
        max_x = min(w, max(x1,x2,x3,x4))
        max_y = min(h, max(y1,y2,y3,y4))
        
        tw = int(np.sqrt((x1-x2)**2 + (y1-y2)**2))
        th = int(np.sqrt((x1-x4)**2 + (y1-y4)**2))
        pt1 = np.float32([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])
        pt2 = np.float32([[0, 0],
                            [tw - 1, 0],
                            [tw - 1, th - 1],
                            [0, th - 1]])
        matrix = cv2.getPerspectiveTransform(pt1,pt2)
        cropped = cv2.warpPerspective(img, matrix, (tw, th)) 
        
        img_list.append(cropped)

    return boxes, img_list