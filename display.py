# import keras
import glob

import keras

# import keras_retinanet
from skimage import exposure

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import configparser
import keras
from keras_retinanet.utils.anchors import AnchorParameters

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


def read_config_file(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def parse_anchor_parameters(config):
    ratios  = np.array(list(map(float, config['anchor_parameters']['ratios'].split(' '))), keras.backend.floatx())
    scales  = np.array(list(map(float, config['anchor_parameters']['scales'].split(' '))), keras.backend.floatx())
    sizes   = list(map(int, config['anchor_parameters']['sizes'].split(' ')))
    strides = list(map(int, config['anchor_parameters']['strides'].split(' ')))
    return AnchorParameters(sizes, strides, ratios, scales)


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('.', 'snapshots/gtsdb/resnet50-gtsdb/', 'resnet50_csv_20_old.h5')


backbone = "resnet50"
# load retinanet model
model = models.load_model(model_path, backbone_name=backbone)

# Convert to inference model
config = read_config_file("/home/deos/e.hrustic/PycharmProjects/keras-retinanet/keras_retinanet/bin/config.ini")
anchor_params = parse_anchor_parameters(config)
model = models.convert_model(model, anchor_params=anchor_params)

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
# model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'speed limit 20', 1: 'speed limit 30', 2: 'speed limit 50', 3: 'speed limit 60', 4: 'speed '
                                                                                                          'limit 70',
                   5: 'speed limit 80', 6: 'end speed limit 80', 7: 'speed limit 100', 8: 'speed limit 120',
                   9: 'no overtaking ', 10: 'no overtaking trucks', 11: 'priority at next intersection',
                   12: 'priority road', 13: 'give way', 14: 'stop', 15: 'no traffic both ways', 16: 'no trucks',
                   17: 'no entry', 18: 'danger', 19: 'bend left', 20: 'bend right', 21: 'bend', 22: 'uneven road',
                   23: 'slippery road', 24: 'road narrows', 25: 'construction', 26: 'traffic signal', 27: 'pedestrian '
                                                                                                          'crossing',
                   28: 'school crossing', 29: 'cycles crossing', 30: 'snow', 31: 'animals', 32: 'restriction ends',
                   33: 'go right', 34: 'go left', 35: 'go straight', 36: 'go right or straight', 37: 'go left or '
                                                                                                     'straight',
                   38: 'keep right', 39: 'keep left', 40: 'roundabout', 41: 'end overtaking', 42: 'end overtaking '
                                                                                                  'trucks'}

# labels_to_names = {1: "A11", 2: "A13", 3: "A14", 4: "A15", 5: "A17", 6: "A19", 7: "A1A", 8: "A1B", 9: "A1C", 10: "A1D", 11: "A21", 12: "A23_geel", 12: "A23", 13: "A25", 14: "A27", 15: "A29", 16: "A3", 17: "A31", 18: "A33", 19: "A35", 20: "A37", 21: "A39", 22: "A41", 23: "A43", 24: "A49", 25: "A5", 26: "A51", 27: "A7A", 28: "A7B", 29: "A7C", 30: "A9", 31: "B1", 32: "B11", 33: "B13", 34: "B15A", 35: "B17", 36: "B19", 37: "B21", 38: "B3", 39: "B5", 40: "B7", 41: "B9", 42: "C1", 43: "C11", 44: "C13", 45: "C15", 46: "C17", 47: "C19", 48: "C21", 49: "C22", 50: "C23", 51: "C24a", 52: "C24b", 53: "C24c", 54: "C25", 55: "C27", 56: "C29", 57: "C3", 58: "C31LEFT", 59: "C31RIGHT", 60: "C33", 61: "C35", 62: "C37", 63: "C39", 64: "C41", 65: "C43", 66: "C45", 67: "C47n", 68: "C48", 69: "C5", 70: "C7", 71: "C9", 72: "D10", 73: "D11", 74: "D13", 75: "D1a", 76: "Db1_schuin_rechts", 76: "D1b_schuin_rechts", 76: "D1b_rechts", 76: "D1b_rechts_onder", 76: "D1b_schuin_links", 76: "D1b", 77: "D1e", 78: "D3b", 79: "D5", 80: "D7", 81: "D9", 82: "E1", 83: "E11", 84: "E3", 85: "E5", 86: "E7", 87: "E9a", 88: "E9a_bewoners", 89: "E9a_disk", 90: "E9a_miva", 91: "E9ag7dn-2", 92: "E9ag7dn-3", 93: "E9ag7dn", 94: "E9b", 94: "X11", 95: "E9c", 96: "E9d", 97: "E9e", 98: "E9f", 99: "E9g", 100: "E9h", 101: "E9i", 102: "F1", 103: "F101a", 104: "F101b", 105: "F101c", 106: "F103n", 107: "F105n", 108: "F107", 109: "F109", 110: "F11", 111: "F12a", 112: "F12b", 113: "F13", 114: "F14", 115: "F15", 116: "F17", 117: "F18", 118: "F19", 119: "F1a_h", 120: "F1a_v", 121: "F1b_h", 122: "F1b_v", 123: "F21", 124: "F23A", 125: "F23B", 126: "F23C", 127: "F23D", 128: "F25", 129: "F27", 130: "F29", 131: "F31", 132: "F33B", 133: "F33C", 134: "F33a", 135: "F34A", 136: "F34B1", 137: "F34B2", 138: "F34C1", 139: "F34C2", 140: "F35", 141: "F37", 142: "F39", 143: "F3a_h", 144: "F3a_v", 145: "F3b_h", 146: "F3b_v", 147: "F41", 148: "F43", 149: "F45", 150: "F47", 151: "F49", 152: "F4a", 153: "F4b", 154: "F5", 155: "F50", 156: "F50bis", 157: "F51A", 158: "F51B", 159: "F53", 160: "F55", 161: "F56", 162: "F57", 163: "F59_links", 163: "F59", 164: "F60", 165: "F61", 166: "F62", 167: "F63", 168: "F65", 169: "F67", 170: "F69", 171: "F7", 172: "F71", 173: "F73", 174: "F75", 175: "F77", 176: "F79", 177: "F8", 178: "F81", 179: "F83", 180: "F85", 181: "F87", 182: "F89", 183: "F9", 184: "F91", 185: "F93", 186: "F95", 187: "F97B", 188: "F98", 189: "F98onder", 190: "F99a", 191: "F99b", 192: "F99c", 193: "Handic", 194: "begin", 195: "betalend", 196: "i-betalend", 197: "bewoners", 198: "disk", 199: "e0c", 200: "einde", 201: "lang", 202: "m1", 203: "m2", 204: "m3", 205: "m4", 206: "m5", 207: "m6", 208: "m7", 209: "m8", 210: "typeII"}

# labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# load image
root_dir = '/home/deos/e.hrustic/Bureau/Datasets/GTSD/TestIJCNN2013/'
# root_dir = '/home/deos/e.hrustic/Bureau/Datasets/BTSD/'
all_img_paths = glob.glob(os.path.join(root_dir, '*.ppm'))
for img in all_img_paths :
    image = read_image_bgr(img) #00031.ppm

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))


    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()