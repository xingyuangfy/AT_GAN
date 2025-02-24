import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
from pdb import set_trace as st


# 定义支持的图像扩展名
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']


def is_image_file(filename):
    """
    检查文件是否为图像文件。
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def list_folder_images(dir, opt):
    """
    列出目录中的图像文件及其对应的解析图路径。
    """
    images = []
    parsings = []
    assert os.path.isdir(dir), f"{dir} is not a valid directory"

    for fname in os.listdir(dir):
        if is_image_file(fname):
            path = os.path.join(dir, fname)
            parsing_fname = fname[:-3] + 'png'  # 解析图文件名（必须是png格式）
            parsing_path = os.path.join(dir, 'parsings', parsing_fname)

            if os.path.isfile(parsing_path):
                images.append(path)
                parsings.append(parsing_path)

    # 如果是FGNET数据集，按身份排序（区分大小写）
    if 'fgnet' in opt.dataroot.lower():
        images.sort(key=str.lower)
        parsings.sort(key=str.lower)

    return images, parsings


def get_transform(opt, normalize=True):
    """
    根据选项生成图像转换。
    """
    transform_list = []

    # 处理图像大小调整和裁剪
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, interpolation=Image.NEAREST))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    # 如果是训练模式且未禁用翻转，则添加随机水平翻转
    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    # 转换为张量
    transform_list.append(transforms.ToTensor())

    # 如果需要归一化，则添加归一化操作
    if normalize:
        mean = (0.5,)
        std = (0.5,)
        transform_list.append(transforms.Normalize(mean, std))

    return transforms.Compose(transform_list)