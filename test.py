# import cv2
# import imutils
# from backend.segmentation.segmentator import Segmentator
# from backend.aligment.scan import DocScanner
# from backend.detection.detector import Detector
# from backend.recognition.recognizer import OCR

# BORDER_SIZE = 20
# RESCALED_HEIGHT = 512


# ss = Segmentator()
# scanner = DocScanner()
# dtt = Detector()
# reg = OCR()

# img = cv2.imread('images/37.jpg')
# ori = img.copy()

# cv2.imshow('ori', ori)

# img_pad = cv2.copyMakeBorder(img, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, cv2.BORDER_CONSTANT, (0,0,0))
# ori = cv2.copyMakeBorder(ori, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, cv2.BORDER_CONSTANT, (0,0,0))

# img_pad = imutils.resize(img_pad, height = int(RESCALED_HEIGHT))

# mask = ss.remove_background(img_pad)
# ali = scanner.scan(ori, mask, binary=False)

# detected = dtt.predict(ali)
# texts = reg.predict_folder(detected[0])
# print(texts)

# cv2.imshow('mask', mask)
# cv2.imshow('ali', ali)
# cv2.imshow('box', detected[2])
# cv2.waitKey(0)

from pipeline import PIPELINE
import cv2 
import numpy as np
from paddleocr import draw_ocr

img = cv2.imread('images/1.jpg')
reader = PIPELINE()     
pd_results, text = reader.infer(img)

ratio = img.shape[0]/512
bbox = np.array(pd_results[1])
bbox = bbox * ratio

# bbox_img = draw_ocr(img, boxes=bbox)
# cv2.imshow('hhahha', bbox_img)
# cv2.waitKey(0)

for item in bbox:
    bbox = {
            'x': item[0][0],
            'y': item[0][1],
            'width': item[1][0] - item[0][0],
            'height': item[3][1] - item[0][1],
            'rotation': 0
        }
    
    print(bbox)