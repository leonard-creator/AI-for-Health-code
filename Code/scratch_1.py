import torch
import kornia as K
import numpy as np
import torchvision
import matplotlib.pyplot as plt

import urllib.request


def show_image(img: torch.Tensor, size: [int, int]):
    out: torch.Tensor = torchvision.utils.make_grid(img, nrow=2, padding=1)
    img_np: np.ndarray = K.tensor_to_image(out)

    plt.figure(figsize=size)
    plt.imshow(img_np)
    plt.axis('off')

#import bilder
import cv2

image: np.ndarray = cv2.imread('/data/project/retina/RIADD/all_cropped_margins/RIADD2.png') #maybe cv2.imread('/data/project/retina/RIADD/all_cropped_margins/RIADD2.png', 1) ==cv2.IMREAD_COLOR('...')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img: torch.Tensor = K.image_to_tensor(image, keepdim=False)
img = img.float() / 255.

show_image(img, (15, 15))

# solarize
aug = K.augmentation.RandomSolarize(0.3, 0.3, p=1.)

img_o: torch.Tensor = aug(img)

show_image(img_o, (8, 8))

# center crop
aug_1 = K.augmentation.CenterCrop(size=(150, 150), p=1., keepdim=True)

img_1: torch.Tensor = aug_1(img)

show_image(img_1, (8, 8))

# color jitter
aug_2 = K.augmentation.ColorJitter(0.3, 0.1, 0.5, 0.01, p=1.)

img_2: torch.Tensor = aug_2(img)

show_image(img_2, (8, 8))


#blur # p=.2
aug_3 = K.augmentation.GaussianBlur((15, 15), (2.0, 2.0))

img_3 : torch.Tensor  = aug_3(img)

show_image(img_3, (15,15))


#horizontal flip
aug_4 = K.augmentation.RandomHorizontalFlip(p=1.0, return_transform=False)

img_4 : torch.Tensor = aug_4(img)
show_image(img_4 , (8,8))

#vertical flip
aug_5 = K.augmentation.RandomVerticalFlip(p=1.0, return_transform=False)

img_5 : torch.Tensor = aug_5(img)
show_image(img_5 , (8,8))

#Affine
aug_6 = K.augmentation.RandomAffine((-15., 25.), p=1.)
img_6: torch.Tensor = aug_6(img)
show_image(img_6 , (8,8))


#random crop
aug_7 = K.augmentation.RandomCrop((2, 2), p=1.)
img_7: torch.Tensor = aug_7(img)
show_image(img_7 , (8,8))

#gray scale
aug_8 = K.augmentation.RandomGrayscale(p=1.0)
img_8: torch.Tensor = aug_8(img)
show_image(img_8 , (8,8))


#perspective
aug_9 = K.augmentation.RandomPerspective(0.6, p=1.)
img_9: torch.Tensor = aug_9(img)
show_image(img_9 , (8,8))

#resized crop
aug_10 = K.augmentation.RandomResizedCrop(size=(100, 100), p=1.)
img_10: torch.Tensor = aug_10(img)
show_image(img_10 , (8,8))


#rotation
aug_11 = K.augmentation.RandomRotation(degrees=180.0, p=1.)
img_11: torch.Tensor = aug_11(img)
show_image(img_11 , (8,8))

#posterize
aug_12 = K.augmentation.RandomPosterize(0, p=1.)
img_12: torch.Tensor = aug_12(img)
show_image(img_12 , (8,8))

#sharpness
aug_13 = K.augmentation.RandomSharpness(.5, p=1.)
img_13: torch.Tensor = aug_13(img)
show_image(img_13 , (10,10))

# normalize
aug_14 = K.augmentation.Normalize(
    mean=torch.Tensor([0.5406, 0.3311, 0.1928]),
    std=torch.Tensor([0.2499, 0.1640, 0.1260]))

img_14: torch.Tensor = aug_14(img)
show_image(img_14, (8, 8))
