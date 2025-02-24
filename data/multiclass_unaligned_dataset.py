"""
Multi-class unaligned dataset module for AT-GAN
Copyright (c) Xingyuangfy 2025. All rights reserved.

This module implements a dataset class for handling unaligned multi-class images,
specifically designed for age transformation tasks.
"""

import os
import re
import torch
import random
import numpy as np
from data.base_dataset import BaseDataset
from data.dataset_utils import list_folder_images, get_transform
from util.preprocess_itw_im import preprocessInTheWildImage
from PIL import Image
from pdb import set_trace as st


# Define age class upper bounds
CLASSES_UPPER_BOUNDS = [2, 6, 9, 14, 19, 29, 39, 49, 69, 120]


class MulticlassUnalignedDataset(BaseDataset):
    def initialize(self, opt):
        """
        Initialize the dataset.

        Args:
            opt: Configuration object containing dataset parameters
        """
        self.opt = opt
        self.root = opt.dataroot
        self.in_the_wild = opt.in_the_wild if not opt.isTrain else False
        self.classNames = []
        self.name_mapping = {}
        self.active_classes_mapping = {}
        self.numClasses = 0
        self.get_samples = False
        self.class_counter = 0
        self.img_counter = 0

        self._initialize_classes()
        self._initialize_directories()
        self._initialize_transform()

        if not self.opt.isTrain:
            self.opt.batchSize = self.numClasses
            self.opt.dataset_size = self.__len__()

    def _initialize_classes(self):
        """
        Initialize class information and mappings.
        Sets up class names and their corresponding age group mappings.
        """
        if not self.in_the_wild:
            subDirs = next(os.walk(self.root))[1]
            prefix = 'train' if self.opt.isTrain else 'test'
            tempClassNames = [currDir[len(prefix):] for currDir in subDirs if prefix in currDir]

            if self.opt.sort_order:
                self.classNames = [name for name in self.opt.sort_order if name in tempClassNames]
            else:
                self.classNames = sorted(tempClassNames)
        else:
            self.classNames = self.opt.sort_order

        for i, name in enumerate(self.classNames):
            self.name_mapping[name] = self._assign_age_class(name)
            self.active_classes_mapping[i] = self.name_mapping[name]

        self.numClasses = len(self.classNames)
        self.opt.numClasses = self.numClasses
        self.opt.classNames = self.classNames

    def _assign_age_class(self, class_name):
        """
        Assign age class based on class name.

        Args:
            class_name: Name of the class containing age information

        Returns:
            int: Index of the assigned age group
        """
        ages = [int(s) for s in re.split('-|_', class_name) if s.isdigit()]
        max_age = ages[-1]
        for i, upper_bound in enumerate(CLASSES_UPPER_BOUNDS):
            if max_age <= upper_bound:
                return i
        return len(CLASSES_UPPER_BOUNDS) - 1

    def _initialize_directories(self):
        """
        Initialize directories and image paths.
        Sets up paths for images and their corresponding parsing files.
        """
        if not self.in_the_wild:
            self.dirs = []
            self.img_paths = []
            self.parsing_paths = []
            self.sizes = []

            for currClass in self.classNames:
                dir_path = os.path.join(self.root, self.opt.phase + currClass)
                imgs, parsings = list_folder_images(dir_path, self.opt)
                self.dirs.append(dir_path)
                self.img_paths.append(imgs)
                self.parsing_paths.append(parsings)
                self.sizes.append(len(imgs))

    def _initialize_transform(self):
        """
        Initialize image transformations.
        Sets up image preprocessing transformations and wild image preprocessor if needed.
        """
        self.transform = get_transform(self.opt)
        if not self.opt.isTrain and self.in_the_wild:
            self.preprocessor = preprocessInTheWildImage(out_size=self.opt.fineSize)

    def set_sample_mode(self, mode=False):
        """
        Set sampling mode for dataset iteration.

        Args:
            mode: Boolean flag for sample mode
        """
        self.get_samples = mode
        self.class_counter = 0
        self.img_counter = 0

    def mask_image(self, img, parsings):
        """
        Mask image based on parsing map.

        Args:
            img: Input image array
            parsings: Parsing map array

        Returns:
            numpy.ndarray: Masked image array
        """
        labels_to_mask = [0, 14, 15, 16, 18]
        for idx in labels_to_mask:
            img[parsings == idx] = 128
        return img

    def get_item_from_path(self, path):
        """
        Load and process image from given path.

        Args:
            path: Path to the image file

        Returns:
            dict: Dictionary containing processed image and metadata
        """
        path_dir, im_name = os.path.split(path)
        img = Image.open(path).convert('RGB')
        img = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[1], img.size[0], 3)

        if self.in_the_wild:
            img, parsing = self.preprocessor.forward(img)
        else:
            parsing_path = os.path.join(path_dir, 'parsings', im_name[:-4] + '.png')
            parsing = Image.open(parsing_path).convert('RGB')
            parsing = np.array(parsing.getdata(), dtype=np.uint8).reshape(parsing.size[1], parsing.size[0], 3)

        img = Image.fromarray(self.mask_image(img, parsing))
        img = self.transform(img).unsqueeze(0)

        return {'Imgs': img, 'Paths': [path], 'Classes': torch.zeros(1, dtype=torch.int), 'Valid': True}

    def __getitem__(self, index):
        """
        Get a data item by index.

        Args:
            index: Index of the data item

        Returns:
            dict: Dictionary containing the data item
        """
        if self.opt.isTrain and not self.get_samples:
            return self._get_train_item(index)
        else:
            return self._get_test_item(index)

    def _get_train_item(self, index):
        """
        Get a training data item.

        Args:
            index: Index of the data item

        Returns:
            dict: Dictionary containing the training pair
        """
        class_A_idx, class_B_idx = self._random_class_indices()
        index_A, index_B = self._random_image_indices(class_A_idx, class_B_idx)

        A_img_path = self.img_paths[class_A_idx][index_A]
        B_img_path = self.img_paths[class_B_idx][index_B]

        A_img, A_parsing = self._load_image_and_parsing(A_img_path)
        B_img, B_parsing = self._load_image_and_parsing(B_img_path)

        A_img = Image.fromarray(self.mask_image(A_img, A_parsing))
        B_img = Image.fromarray(self.mask_image(B_img, B_parsing))

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        return {'A': A_img, 'B': B_img,
                "A_class": class_A_idx, "B_class": class_B_idx,
                'A_paths': A_img_path, 'B_paths': B_img_path}

    def _random_class_indices(self):
        """
        Generate random indices for two different classes.

        Returns:
            tuple: Two different class indices
        """
        class_A_idx = random.randint(0, self.numClasses - 1)
        class_B_idx = random.randint(0, self.numClasses - 1)
        while class_A_idx == class_B_idx:
            class_B_idx = random.randint(0, self.numClasses - 1)
        return class_A_idx, class_B_idx

    def _random_image_indices(self, class_A_idx, class_B_idx):
        """
        Generate random image indices for two classes.

        Args:
            class_A_idx: Index of first class
            class_B_idx: Index of second class

        Returns:
            tuple: Two image indices
        """
        index_A = random.randint(0, self.sizes[class_A_idx] - 1)
        index_B = random.randint(0, self.sizes[class_B_idx] - 1)
        return index_A, index_B

    def _load_image_and_parsing(self, img_path):
        """
        Load image and its parsing map.

        Args:
            img_path: Path to the image file

        Returns:
            tuple: Image array and parsing map array
        """
        img = Image.open(img_path).convert('RGB')
        img = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[1], img.size[0], 3)

        if self.in_the_wild:
            img, parsing = self.preprocessor.forward(img)
        else:
            path_dir, im_name = os.path.split(img_path)
            parsing_path = os.path.join(path_dir, 'parsings', im_name[:-4] + '.png')
            parsing = Image.open(parsing_path).convert('RGB')
            parsing = np.array(parsing.getdata(), dtype=np.uint8).reshape(parsing.size[1], parsing.size[0], 3)
        return img, parsing

    def _get_test_item(self, index):
        """
        Get a test data item.

        Args:
            index: Index of the data item

        Returns:
            dict: Dictionary containing the test item
        """
        class_idx = self.class_counter % self.numClasses
        self.class_counter += 1

        if self.get_samples:
            ind = random.randint(0, self.sizes[class_idx] - 1)
        else:
            ind = self.img_counter if self.img_counter < self.sizes[class_idx] else -1

        if ind > -1:
            img_path = self.img_paths[class_idx][ind]
            img, parsing = self._load_image_and_parsing(img_path)
            img = Image.fromarray(self.mask_image(img, parsing))
            img = self.transform(img)
            valid = True
        else:
            img = torch.zeros(3, self.opt.fineSize, self.opt.fineSize)
            img_path = ''
            valid = False

        if class_idx == self.numClasses - 1:
            self.img_counter += 1

        return {'Imgs': img, 'Paths': img_path, 'Classes': class_idx, 'Valid': valid}

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Number of items in the dataset
        """
        if self.opt.isTrain:
            return round(sum(self.sizes) / 2)
        elif self.in_the_wild:
            return 0
        else:
            return max(self.sizes) * self.numClasses

    def name(self):
        """
        Get the name of the dataset.

        Returns:
            str: Dataset name
        """
        return 'MulticlassUnalignedDataset'
