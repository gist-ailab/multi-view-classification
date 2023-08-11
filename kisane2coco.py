import os
import cv2
import json
import h5py
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

class kisane2coco():
    def __init__(self, mode):
        self.image_dict_list = []
        self.annot_dict_list = []
        self.categ_dict_list = []
        for i in range(0, 200):
            self.categ_dict_list.append({
                'id': i,
                'name': 'no_name'
            })

        if mode == 'train':
            self.is_train = True
        else:
            self.is_train = False

        path = '/ailab_mat/dataset/kisane_DB/kisane_DB/single_data/'
        self.base_folder = [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]


    def make_annot(self):
        annot_idx = 0
        image_idx = 0
        class_idx = 0

        for folder in self.base_folder:
            folder_tray = os.path.join(folder, 'TRAY1')
            folder_pt = os.listdir(folder_tray)
            folders_tp = os.listdir(os.path.join(folder, 'TRAY1', folder_pt[0]))

            for tp in folders_tp:
                if self.is_train:
                    if tp.endswith(('7', '8', '9')):
                        continue
                else:
                    if tp.endswith(('1', '2', '3', '4', '5', '6')):
                        continue
                tp_folder = os.path.join(folder, 'TRAY1', folder_pt[0], tp)
                for img_folder in os.listdir(tp_folder):
                    final_img_folder = os.path.join(tp_folder, img_folder)
                    base_file_names = [x[:-4] for x in os.listdir(final_img_folder) if x.endswith('.txt')]

                    for base_file_name in base_file_names:
                        img_file_name = base_file_name + '_Color.png'
                        csv_file_name = base_file_name + '_GT.csv'

                        if not os.path.exists(os.path.join(final_img_folder, img_file_name)) \
                        or not os.path.exists(os.path.join(final_img_folder, csv_file_name)):
                            continue

                        cls, bbox = self.csv_read(os.path.join(final_img_folder, csv_file_name))

                        self.image_dict_list.append({
                            'id': image_idx,
                            'image_id': image_idx,
                            'file_name': os.path.join(final_img_folder, img_file_name),
                            'width': 1280,
                            'height': 720,
                        })

                        self.annot_dict_list.append({
                            'id': annot_idx,
                            'image_id': image_idx,
                            'category_id': class_idx,
                            'width': 1280,
                            'height': 720,
                            'bbox': bbox,
                        })

                        image_idx += 1
                        annot_idx += 1
            class_idx += 1
 

    def csv_read(self, csv_path):
        with open(csv_path, 'r') as f:
            info = f.readlines()[1].split(',')
            cls = int(info[5])
            bbox = [int(info[1]), int(info[2]), int(info[3]), int(info[4])]
        return cls, bbox



    def combine_annot(self):
        self.make_annot()
        print('image len', len(self.image_dict_list))
        print('annot len', len(self.annot_dict_list))

        return {
            'images': self.image_dict_list,
            'annotations': self.annot_dict_list,
            'categories': self.categ_dict_list
        }





if __name__ == '__main__':

    print('train')
    train_converter = kisane2coco(mode='train')
    train_annot = train_converter.combine_annot()

    with open('kisane_single_train.json', 'w') as f:
        json.dump(train_annot, f)

    print('val')
    val_converter = kisane2coco(mode='val')
    val_annot = val_converter.combine_annot()

    with open('kisane_single_val.json', 'w') as f:
        json.dump(val_annot, f)


