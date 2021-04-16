import torch
import kornia as K
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import cv2


def show_image(img: torch.Tensor, size: [int, int]):
    out: torch.Tensor = torchvision.utils.make_grid(img, nrow=2, padding=1)
    img_np: np.ndarray = K.tensor_to_image(out)

    plt.figure(figsize=size)
    plt.imshow(img_np)
    plt.axis('off')

#einlesen bild
image: np.ndarray = cv2.imread('/data/project/retina/RIADD/all_cropped_margins/RIADD2.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img: torch.Tensor = K.image_to_tensor(image, keepdim=False)
img = img.float() / 255.

# pipeline aus dem repo
transforms_list1 = torch.nn.Sequential(
    K.augmentation.RandomCrop(size=(300, 300),
                              return_transform=False, pad_if_needed=True),
    K.augmentation.ColorJitter(0.4, 0.4, 0.2, 0.1),
    K.augmentation.RandomHorizontalFlip(return_transform=False),
    K.augmentation.RandomVerticalFlip(return_transform=False),
    K.augmentation.GaussianBlur((23, 23), (0.1, 2.0)),
    K.augmentation.CenterCrop(size=(300, 300), p=1., return_transform=False),
    K.augmentation.Normalize(
        mean=torch.Tensor([0.5406, 0.3311, 0.1928]),
        std=torch.Tensor([0.2499, 0.1640, 0.1260])
    )

)
im = transforms_list1(img)
show_image(im, (8,8))



#eigene pipeline_1  mit solarize
transform = torch.nn.Sequential(
        K.augmentation.RandomCrop(size=(300, 300),p=1.,
                                  return_transform=False, pad_if_needed=True),
        K.augmentation.ColorJitter(0.4, 0.4, 0.2, 0.1),
        K.augmentation.RandomHorizontalFlip( p=1.,return_transform=False),
        K.augmentation.RandomVerticalFlip( p=1.,return_transform=False),
        K.augmentation.GaussianBlur((23, 23), (0.1, 2.0)),
        K.augmentation.RandomSolarize(0.5, 0.5, p=1.),
        K.augmentation.CenterCrop(size=(300,300), p=1.,return_transform=False)
)

images = transform(img)
show_image(images, (8,8))


#pipeline 2 mit grayscale
transform_2 = torch.nn.Sequential(
        K.augmentation.RandomCrop(size=(300, 300),p=1.,
                                  return_transform=False, pad_if_needed=True),
        K.augmentation.ColorJitter(0.4, 0.4, 0.2, 0.1),
        K.augmentation.RandomHorizontalFlip( p=1.,return_transform=False),
        K.augmentation.RandomVerticalFlip( p=1.,return_transform=False),
        K.augmentation.RandomGrayscale(p=1.),
        K.augmentation.GaussianBlur((23, 23), (0.1, 2.0)),
        K.augmentation.RandomSolarize(0.5, 0.5, p=1.),
        K.augmentation.CenterCrop(size=(300,300), p=1.,return_transform=False)
)

image_2 = transform_2(img)
show_image(image_2, (8,8))

# pipeline 3 mit nomalinsation
transform_3 = torch.nn.Sequential(
    K.augmentation.RandomCrop(size=(300, 300), p=1.,
                              return_transform=False, pad_if_needed=True),
    K.augmentation.ColorJitter(0.4, 0.4, 0.2, 0.1),
    K.augmentation.RandomHorizontalFlip(p=1., return_transform=False),
    K.augmentation.RandomVerticalFlip(p=1., return_transform=False),
    K.augmentation.RandomGrayscale(p=1.),
    K.augmentation.GaussianBlur((23, 23), (0.1, 2.0)),
    K.augmentation.RandomSolarize(0.5, 0.5, p=1.),
    K.augmentation.CenterCrop(size=(300, 300), p=1., return_transform=False),
    K.augmentation.Normalize(
        mean=torch.Tensor([0.5406, 0.3311, 0.1928]),
        std=torch.Tensor([0.2499, 0.1640, 0.1260])
    )

)


image_3 = transform_3(img)
show_image(image_3, (8,8))