import cv2
import os 
import time
from configs import config as cf
from paddleocr import PaddleOCR, draw_ocr
from utils.backend_utils import crop_box


class Detector:
    def __init__(self):
        # self.save_results = cf.box_img_dir
        h_w_ratio = 1               # h/w ratio of padded pic
        lang = 'en'
        det_db_unclip_ratio = 1.75  #default 1.5
        det_limit_side_len = 960    #default 960
        max_text_length = 50
        use_space_char = True
        det_db_box_thresh=0.4       #default 0.6
        drop_score = 0.8
        det_model_dir = cf.text_detection_dir

        self.ocr_model = PaddleOCR(lang=lang,
                            det_db_unclip_ratio=det_db_unclip_ratio,
                            det_limit_side_len=det_limit_side_len,
                            max_text_length=max_text_length,
                            use_space_char=use_space_char,
                            det_db_box_thresh=det_db_box_thresh,
                            drop_score = drop_score,
                            #det_model_dir = det_model_dir                 
                            )

    def predict(self, img):
        bbox = self.ocr_model.ocr(img, rec=False, cls=False)
        boxes, img_list = crop_box(img, bbox)
        img_bbox = draw_ocr(img, boxes= bbox)
        # cv2.imwrite(os.path.join(self.save_results, base_name), img_bbox)
        paddle_results = [img_list, bbox, img_bbox]
        return paddle_results