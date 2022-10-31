import os

device = "cuda"


input_img_dir = os.path.join('images', 'input')
mask_img_dir = os.path.join('results', 'mask')
align_img_dir = os.path.join('results','align')
cropped_img_dir = os.path.join('results', 'drop')



img_dir = "./images"
result_img_dir = "./results/model"
raw_img_dir = "./results/raw"
box_img_dir = "./results/box"

alphabet = ' "%&\'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyzÀÁÂÃÉÊÌÍÒÓÔÙÚÝàáâãèéêìíòóôõùúýĂăĐđĩŨũƠơƯưẠạẢảẤấẦầẨẩẫẬậẮắẰằẳẶặẹẺẻẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỶỷỸỹ'
node_labels = ['OTHER', 'ADDRESS', 'SELLER', 'TIMESTAMP', 'TOTAL_COST']

text_detection_dir = "weights/text_detect"
saliency_weight_path = "./weights/segment/u2netp.pth"
kie_weight_path = "./weights/kie/kie_mcocr.pkl"

text_reg_config = "weights/text_recognition/vgg-seq2seq.yml"
text_reg_model = "weights/text_recognition/vggseq2seq.pth"

# 0.5, 0.82
saliency_ths = 0.5
score_ths = 0.82
get_max = True  # get max score / filter predicted categories by score threshold
merge_text = True  # if True, concatenate text contents from left to right
infer_batch_vietocr = True  # inference with batch
visualize = False
