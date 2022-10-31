import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from configs import config as cf


class OCR:
    def __init__(self):
        self.config_path = cf.text_reg_config
        self.model_path = cf.text_reg_model
        self.load_config()
        self.detector = Predictor(self.config)
    
    def load_config(self):
        self.config = Cfg.load_config_from_file(self.config_path)
        self.config['weights'] = self.model_path
        self.config['cnn']['pretrained']=False
        self.config['device'] = 'cuda:0'
        self.config['predictor']['beamsearch']=False
    
    def predict_folder(self, img_list):
        texts = []
        for item in img_list: 
            img = Image.fromarray(item)
            s = self.detector.predict(img)
            texts.append(s)
        return texts