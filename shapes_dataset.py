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

import copy

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


def ellipse(
    output_path: str = "ignore", channels_first: bool = True
) -> Tuple[np.array, int]:
    new_colors = copy.deepcopy(COLORS)
    random.shuffle(new_colors)
    bg_color, fill_color = random.sample(set(new_colors), 2)

    # print(bg_color, fill_color)
    image = Image.new("RGB", IMAGE_SIZE, bg_color)
    draw = ImageDraw.Draw(image)

    draw.ellipse((10, 10, 25, 50), fill=fill_color)
    # draw.ellipse((100, 150, 275, 300), outline="black", width=5,
    #              fill="yellow")

    if output_path != "ignore":
        image.save(output_path)

    # print(('ellipse', bg_color, fill_color))
    val = val_per_comb.index(("ellipse", bg_color, fill_color))

    if channels_first:
        return np.moveaxis(np.array(image), -1, 0), val
    return np.array(image), val


def circle(
    output_path: str = "ignore", channels_first: bool = True
) -> Tuple[np.array, int]:
    new_colors = copy.deepcopy(COLORS)
    random.shuffle(new_colors)
    bg_color, fill_color = random.sample(new_colors, 2)

    image = Image.new("RGB", IMAGE_SIZE, bg_color)
    draw = ImageDraw.Draw(image)

    draw.ellipse((10, 10, 50, 50), fill=fill_color)

    if output_path != "ignore":
        image.save(output_path)

    # print(("circle", bg_color, fill_color))
    val = val_per_comb.index(("circle", bg_color, fill_color))

    if channels_first:
        return np.moveaxis(np.array(image), -1, 0), val
    return np.array(image), val


def pentagon(
    output_path: str = "ignore", channels_first: bool = True
) -> Tuple[np.array, int]:
    new_colors = copy.deepcopy(COLORS)
    random.shuffle(new_colors)
    bg_color, fill_color = random.sample(new_colors, 2)

    image = Image.new("RGB", IMAGE_SIZE, bg_color)
    draw = ImageDraw.Draw(image)

    draw.regular_polygon((32, 32, 20), 5, fill=fill_color)
    if output_path != "ignore":
        image.save(output_path)

    # print(("pentagon", bg_color, fill_color))
    val = val_per_comb.index(("pentagon", bg_color, fill_color))

    if channels_first:
        return np.moveaxis(np.array(image), -1, 0), val
    return np.array(image), val


def rectangle(
    output_path: str = "ignore", channels_first: bool = True
) -> Tuple[np.array, int]:
    new_colors = copy.deepcopy(COLORS)
    random.shuffle(new_colors)
    bg_color, fill_color = random.sample(new_colors, 2)

    image = Image.new("RGB", IMAGE_SIZE, bg_color)
    draw = ImageDraw.Draw(image)

    draw.rectangle((10, 20, 40, 50), fill=fill_color)
    if output_path != "ignore":
        image.save(output_path)

    # print(("rectangle", bg_color, fill_color))
    val = val_per_comb.index(("rectangle", bg_color, fill_color))

    if channels_first:
        return np.moveaxis(np.array(image), -1, 0), val
    return np.array(image), val


class ShapesSummation(Dataset):
    def __init__(
        self,
        min_len: int,
        max_len: int,
        dataset_len: int,
    ):
        self.min_len = min_len
        self.max_len = max_len
        self.dataset_len = dataset_len

        self.items_len_range = list(range(self.min_len, self.max_len + 1))

        self.shape_funcs = [ellipse, circle, pentagon, rectangle]

    def __len__(self) -> int:
        return self.dataset_len

    def torchify(self, x: np.array, norm: bool = True) -> torch.tensor:
        x = torch.tensor(x)
        if norm:
            return x / 255.0
        return x

    def __getitem__(self, item: int) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:

        set_size = random.choice(self.items_len_range)
        the_sum = 0
        images = []
        targets = []
        for _ in range(set_size):
            img, target = self.shape_funcs[random.randint(0, 3)]()
            img = self.torchify(img)
            the_sum += target
            images.append(img)
            targets.append(target)

        return (
            torch.stack(images, dim=0),
            torch.tensor(targets),
            torch.FloatTensor([the_sum]),
        )


class VisualSimOracle(Dataset):
    def __init__(
        self,
        dataset_len: int,
    ):
        """
        Return a tuple of two random (x, y) and 1 if x = y, or -1 if x != y
        """
        self.dataset_len = dataset_len
        self.shape_funcs = [ellipse, circle, pentagon, rectangle]

    def __len__(self) -> int:
        return self.dataset_len

    def torchify(self, x: np.array, norm: bool = True) -> torch.tensor:
        x = torch.tensor(x)
        if norm:
            return x / 255.0
        return x

    def __getitem__(self, item: int) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:

        x_img, x_target = self.shape_funcs[random.randint(0, 3)]()
        x_img = self.torchify(x_img)

        y_img, y_target = self.shape_funcs[random.randint(0, 3)]()
        y_img = self.torchify(y_img)

        indicator = 1 if x_target == y_target else -1

        return x_img, y_img, indicator


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # print(ellipse("assets/ellipse.png"))
    # print(pentagon("assets/pentagon.png"))
    # print(circle("assets/circle.png"))
    # print(rectangle("assets/rectangle.png"))

    # print(list(zip(val_per_comb, list(range(len(val_per_comb))))))

    shapes = ShapesSummation(2, 10, 100)
    stack, tar, su = shapes.__getitem__(0)
    print(stack.shape)

    oracle = VisualSimOracle(100)
    x, y, i = oracle.__getitem__(0)
    # print(x.shape, i)

    oracle_dataloader = DataLoader(oracle, batch_size=5)
    b_x, b_y, b_i = next(iter(oracle_dataloader))
    print(b_i)
