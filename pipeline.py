from tkinter import image_names
import imutils
import cv2
from backend.segmentation.segmentator import Segmentator
from backend.aligment.scan import DocScanner
from backend.detection.detector import Detector
from backend.recognition.recognizer import OCR

class PIPELINE:
    def __init__(self):
        pass
        # Load segment model 
        self.segmentator = Segmentator()
        # Load scanner
        self.scanner = DocScanner()
        # Load detection model
        self.detector = Detector()
        # Load recognize model
        self.recognizer = OCR()
        # Set up preprocess parameters
        self.BORDER_SIZE = 20
        self.RESCALED_HEIGHT = 512
    

    def infer(self, image):
        # Resize image
        #ori_pad = cv2.copyMakeBorder(image, self.BORDER_SIZE, self.BORDER_SIZE, self.BORDER_SIZE, self.BORDER_SIZE, cv2.BORDER_CONSTANT, (0,0,0))
        #img_pad = cv2.copyMakeBorder(image, self.BORDER_SIZE, self.BORDER_SIZE, self.BORDER_SIZE, self.BORDER_SIZE, cv2.BORDER_CONSTANT, (0,0,0))
        img_pad = imutils.resize(image, height = int(self.RESCALED_HEIGHT))


        '''
            IMAGE SEGMENTATION
            --input: image (Opencv Object)
            --outpt: image (Opencv Object)
            --description: Remove background and return a binary image(var) with the main object is the recipt
        '''
        #masking = self.segmentator.remove_background(img_pad)


        '''
            ALIGN IAMGE
            --input: image (Opencv Object)
            --outpt: image (Opencv Object)
            --description: Find the contours and warp images(var)
        '''
        #aligned_image = self.scanner.scan(ori_pad, masking)


        '''
            TEXT DETCTION
            --input: image
            --output: coordinate of the text boxes
            --description: Receive an image and find text boxes(var)
        '''
        detected_results = self.detector.predict(img_pad)


        '''
            TEXT RECOGNITIONS
            --input: image
            --output: texts 
            --description: Input an images and output the text of the image
        '''
        texts = self.recognizer.predict_folder(detected_results[0])


        # cv2.imshow('ok chua', detected_results[2])
        # cv2.waitKey(0)
        # print(texts)

        del img_pad

        return detected_results, texts