from glob import glob

from model import Sobel
import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time


def np_img_to_tensor(img):
    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    return img_tensor


def tensor_to_np_img(img_tensor):
    img = img_tensor.cpu().permute(0, 2, 3, 1).numpy()
    return img[0, ...]  # get the first element since it's batch form


def sobel_torch_version(img_np, torch_sobel):
    img_tensor = np_img_to_tensor(np.float32(img_np))
    img_edged = tensor_to_np_img(torch_sobel(img_tensor))
    img_edged = np.squeeze(img_edged)
    return img_edged


def main():
    img_dir = "sample-imgs/*"
    imgs = sorted(glob(img_dir))
    torch_sobel = Sobel()
    for img in imgs:
        rgb_orig = cv2.imread(img)
        rgb_orig = cv2.resize(rgb_orig, (224, 224))
        rgb_edged = sobel_torch_version(rgb_orig, torch_sobel=torch_sobel)

        cv2.imshow("image", rgb_edged)


if __name__ == "__main__":
    main()
