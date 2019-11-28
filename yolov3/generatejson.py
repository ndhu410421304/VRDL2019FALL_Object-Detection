# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from model import yolov3

# Use time to calculate local speed
# json to output predictions
import time
import json

# paser commands: insert from command prompt
parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("input_image", type=str,
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[64, 64],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/data.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)  # load from anchor file
args.classes = read_class_names(args.class_name_path)  #name file
args.num_class = len(args.classes)  # calculate total classes

# color use to visualize result
color_table = get_color_table(args.num_class)

j_data = []
cur_time = time.time()  # start countdown

with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data') # placeholder to load input
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3', reuse=tf.AUTO_REUSE):  #set as reusable variable
        pred_feature_maps = yolo_model.forward(input_data, False)  # feed the data inside model
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs
    # compute result on gpu
    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.5, nms_thresh=0.45)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)
    # oredict bounding box for each image
    # 13k in total
    for i in range(13068):
        img_path = "data/test/" + str(i+1) + ".png"
        print(img_path)
        img_ori = cv2.imread(img_path)
        if args.letterbox_resize:
            img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
        else:
            height_ori, width_ori = img_ori.shape[:2]
            img = cv2.resize(img_ori, tuple(args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.
        with tf.device('/gpu:0'):  # use gpu to calculate prediction
            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
            labels_ = np.where(labels_==0, 10, labels_)  # replace the label from 0 to 10

        # rescale the coordinates to the original image
        if args.letterbox_resize:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
            boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))

        boxes_[:, [0, 2]], boxes_[:, [1, 3]] = boxes_[:, [1, 3]], boxes_[:, [0, 2]]
        boxes_ = boxes_.astype(int)  # change type to integer to match requirment
        
        print("box coords:")
        print(boxes_)
        print('*' * 30)
        print("scores:")
        print(scores_)
        print('*' * 30)
        print("labels:")
        print(labels_)

        # visualize result
        '''
        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
        cv2.imshow('Detection result', img_ori)
        cv2.imwrite('detection_result.jpg', img_ori)
        cv2.waitKey(0)
        '''
        data = {}
        data['bbox'] = boxes_.tolist()
        data['score'] = scores_.tolist()
        data['label'] = labels_.tolist()
        j_data.append(data)

with open('data.json', 'w') as outfile:
    json.dump(j_data, outfile)
# remeber to take the file out!!!
# print approximate time use per image
Ftime = (time.time() - cur_time) / 13068
print("TimeUse: " + str(Ftime))