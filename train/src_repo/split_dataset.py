'''
Usage: python split_dataset.py /home/data
'''
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import shutil
import pathlib
import random

import xml.etree.ElementTree as ET
import io

dataset_root = '/project/train/src_repo/dataset'
supported_fmt = ['.jpg', '.JPG']
       
if __name__ == '__main__':
    os.makedirs(dataset_root, exist_ok=True)
    if not os.path.exists(sys.argv[1]):
        print(f'{sys.argv[1]} 不存在!')
        exit(-1)

    # 遍历数据集目录下所有xml文件及其对应的图片
    dataset_path = pathlib.Path(sys.argv[1])
    found_data_list = []
    for mask_file in dataset_path.glob('**/*.png'):
        possible_images = [mask_file.with_suffix(suffix) for suffix in supported_fmt]
        supported_images = list(filter(lambda p: p.is_file(), possible_images))
        if len(supported_images) == 0:
            print(f'找不到对应的图片文件：`{mask_file.as_posix()}`')
            continue
        found_data_list.append({'image': supported_images[0], 'label': mask_file})

    # 随机化数据集，将数据集拆分成训练集和验证集，并将其拷贝到/project/train/src_repo/dataset下
    random.shuffle(found_data_list)
    train_data_count = len(found_data_list) * 4 / 5
    train_data_list = []
    valid_data_list = []
    for i, data in enumerate(found_data_list):
        if i < train_data_count:  # 训练集
            data_list = train_data_list
        else:  # 验证集
            data_list = valid_data_list
        data_list.append(data)

    with open(os.path.join(dataset_root, 'train.txt'), 'w') as f:
        for name in train_data_list:
            f.write(name['image'].as_posix() + ',' + name['label'].as_posix() + '\n')
    with open(os.path.join(dataset_root, 'val.txt'), 'w') as f:
        for name in valid_data_list:
            f.write(name['image'].as_posix() + ',' + name['label'].as_posix() + '\n')
    print('Done')