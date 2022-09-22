# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random, cv2
import numpy as np
from dataset.base_image_dataset import BaseImageDataset
from data.image_loader import opencv_loader
from admin.environment import env_settings


class ZurichRAW2RGB(BaseImageDataset):
    """
    Dataset class for loading the Canon RGB images from the Zurich RAW 2 RGB dataset [1]

    [1] Replacing Mobile Camera ISP with a Single Deep Learning Model. Andrey Ignatov, Luc Van Gool and Radu Timofte,
        CVPRW 2020
    """
    def __init__(self, root=None, split='train', image_loader=opencv_loader, initialize=True):
        """
        args:
            root - Path to root dataset directory
            split - Dataset split to use. Can be 'train' or 'test'
            image_loader - loader used to read the images
            initialize - boolean indicating whether to load the meta-data for the dataset
        """
        root = env_settings().zurichraw2rgb_dir if root is None else root
        super().__init__('ZurichRAW2RGB', root, image_loader)
        self.split = split

        if initialize:
            self.initialize()

    def initialize(self):
        split = self.split
        root = self.root
        if split in ['train', 'test']:
            self.img_pth = os.path.join(root, split, 'canon')
        else:
            raise Exception('Unknown split {}'.format(split))

        self.image_list = self._get_image_list(split)

    def _get_image_list(self, split):
        if split == 'train':
            image_list = ['{:d}.jpg'.format(i) for i in range(46839)]
        elif split == 'test':
            image_list = ['{:d}.jpg'.format(i) for i in range(1204)]
        else:
            raise Exception

        return image_list

    def get_image_info(self, im_id):
        return {}

    def _get_image(self, im_id):
        path = os.path.join(self.img_pth, self.image_list[im_id])
        img = cv2.imread(path)
        # if random.randint(0,1) == 1 and self.split=='train':
        #     flag_aug = random.randint(1,7)
        #     img = self.data_augmentation(img, flag_aug)
        # else:
        img = img
        return img
    
    def __getitem__(self, index):
        frame = self._get_image(index)

        return frame

    def get_image(self, im_id, info=None):
        frame = self._get_image(im_id)

        if info is None:
            info = self.get_image_info(im_id)

        return frame, info

    def data_augmentation(self, image, mode):
        """
        Performs data augmentation of the input image
        Input:
            image: a cv2 (OpenCV) image
            mode: int. Choice of transformation to apply to the image
                    0 - no transformation
                    1 - flip up and down
                    2 - rotate counterwise 90 degree
                    3 - rotate 90 degree and flip up and down
                    4 - rotate 180 degree
                    5 - rotate 180 degree and flip
                    6 - rotate 270 degree
                    7 - rotate 270 degree and flip
        """
        if mode == 0:
            # original
            out = image
        elif mode == 1:
            # flip up and down
            out = np.flipud(image)
        elif mode == 2:
            # rotate counterwise 90 degree
            out = np.rot90(image)
        elif mode == 3:
            # rotate 90 degree and flip up and down
            out = np.rot90(image)
            out = np.flipud(out)
        elif mode == 4:
            # rotate 180 degree
            out = np.rot90(image, k=2)
        elif mode == 5:
            # rotate 180 degree and flip
            out = np.rot90(image, k=2)
            out = np.flipud(out)
        elif mode == 6:
            # rotate 270 degree
            out = np.rot90(image, k=3)
        elif mode == 7:
            # rotate 270 degree and flip
            out = np.rot90(image, k=3)
            out = np.flipud(out)
        else:
            raise Exception('Invalid choice of image transformation')
        return out.copy()