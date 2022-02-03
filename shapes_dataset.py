from typing import Tuple
from matplotlib.pyplot import fill

import numpy as np
from PIL import Image, ImageDraw
import itertools

import random


import torch
from torch import FloatTensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize


"""
2D Shapes Dataset:

4 Shapes:
1. Circle
2. Rectangle 
3. Pentagon
4. Ellipse 

4 Colors:
1. Red 
2. Green
3. Blue 
4. Yellow

Each image has following possible variations: 

(4 x Shapes) x (4 x Background Color) x (3 x Shape Color) = 48

Background and Shape color will be different

Each has a uniuqe value
"""

IMAGE_SIZE = (64, 64)
COLORS = ["red", "green", "blue", "yellow"]
SHAPES = ["ellipse", "circle", "pentagon", "rectangle"]


def make_sure_unique(comb):
    if comb[1] == comb[2]:
        return False
    return True

val_per_comb = list([r for r in itertools.product(SHAPES, COLORS, COLORS)])
val_per_comb = list(filter(make_sure_unique, val_per_comb))

def ellipse(output_path: str = "ignore") -> Tuple[np.array, int]:
    bg_color, fill_color = random.sample(COLORS, 2)
    image = Image.new("RGB", IMAGE_SIZE, bg_color)
    draw = ImageDraw.Draw(image)

    draw.ellipse((10, 10, 25, 50), fill=fill_color)
    # draw.ellipse((100, 150, 275, 300), outline="black", width=5,
    #              fill="yellow")

    if output_path != "ignore":
        image.save(output_path)

    val = val_per_comb.index(("ellipse", bg_color, fill_color))

    return np.array(image), val


def circle(output_path: str = "ignore") -> Tuple[np.array, int]:

    bg_color, fill_color = random.sample(COLORS, 2)

    image = Image.new("RGB", IMAGE_SIZE, bg_color)
    draw = ImageDraw.Draw(image)

    draw.ellipse((10, 10, 50, 50), fill=fill_color)

    if output_path != "ignore":
        image.save(output_path)

    val = val_per_comb.index(("circle", bg_color, fill_color))

    return np.array(image), val


def pentagon(output_path: str = "ignore") -> Tuple[np.array, int]:
    bg_color, fill_color = random.sample(COLORS, 2)
    image = Image.new("RGB", IMAGE_SIZE, bg_color)
    draw = ImageDraw.Draw(image)

    draw.regular_polygon((32, 32, 20), 5, fill=fill_color)
    if output_path != "ignore":
        image.save(output_path)

    val = val_per_comb.index(("pentagon", bg_color, fill_color))

    return np.array(image), val


def rectangle(output_path: str = "ignore") -> Tuple[np.array, int]:
    bg_color, fill_color = random.sample(COLORS, 2)
    image = Image.new("RGB", IMAGE_SIZE, bg_color)
    draw = ImageDraw.Draw(image)

    draw.rectangle((10, 20, 40, 50), fill=fill_color)
    if output_path != "ignore":
        image.save(output_path)

    val = val_per_comb.index(("rectangle", bg_color, fill_color))

    return np.array(image), val

SHAPES_TRANSFORM = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])



class ShapesSummation(Dataset):
    def __init__(self, min_len: int, max_len: int, dataset_len: int, transform: Compose = SHAPES_TRANSFORM):
        self.min_len = min_len
        self.max_len = max_len
        self.dataset_len = dataset_len
        self.transform = transform

        items_len_range = np.arange(self.min_len, self.max_len + 1)
        items_len = np.random.choice(items_len_range, size=self.dataset_len, replace=True)
        
        

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, item: int) -> Tuple[FloatTensor,FloatTensor, FloatTensor]:
        mnist_items = self.mnist_items[item]

        the_sum = 0
        images = []
        targets = []
        for mi in mnist_items:
            img, target = self.mnist.__getitem__(mi)
            the_sum += target
            images.append(img)
            targets.append(target)

        return torch.stack(images, dim=0), torch.tensor(targets), torch.FloatTensor([the_sum])



if __name__ == "__main__":

    # print(ellipse("assets/ellipse.png"))
    # print(pentagon("assets/pentagon.png"))
    # print(circle("assets/circle.png"))
    # print(rectangle("assets/rectangle.png"))

    print(val_per_comb)
