from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw
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




"""

IMAGE_SIZE = (64, 64)
COLORS = ["red", "green", "blue", "yellow"]


def ellipse(output_path:str = "ignore") -> np.array:
    bg_color , fill_color = random.sample(COLORS, 2)
    image = Image.new("RGB", IMAGE_SIZE, bg_color)
    draw = ImageDraw.Draw(image)

    draw.ellipse((10, 10, 25, 50), fill= fill_color)
    # draw.ellipse((100, 150, 275, 300), outline="black", width=5,
    #              fill="yellow")

    if output_path != "ignore":
        image.save(output_path)

    return np.array(image)


def circle(output_path:str = "ignore") -> np.array:
    bg_color , fill_color = random.sample(COLORS, 2)
    image = Image.new("RGB", IMAGE_SIZE, bg_color)
    draw = ImageDraw.Draw(image)

    draw.ellipse((10, 10, 50, 50), fill= fill_color)
    # draw.ellipse((100, 150, 275, 300), outline="black", width=5,
    #              fill="yellow")

    if output_path != "ignore":
        image.save(output_path)

    return np.array(image)


def pentagon(output_path:str = "ignore") -> np.array:
    bg_color , fill_color = random.sample(COLORS, 2)
    image = Image.new("RGB", IMAGE_SIZE, bg_color)
    draw = ImageDraw.Draw(image)

    draw.regular_polygon((32, 32, 20), 5, fill= fill_color)
    if output_path != "ignore":
        image.save(output_path)

    return np.array(image)

def rectangle(output_path:str = "ignore") -> np.array:
    bg_color , fill_color = random.sample(COLORS, 2)
    image = Image.new("RGB", IMAGE_SIZE, bg_color)
    draw = ImageDraw.Draw(image)

    draw.rectangle((10, 20, 40, 50 ), fill= fill_color)
    if output_path != "ignore":
        image.save(output_path)

    return np.array(image)





if __name__ == "__main__":

print(ellipse("assets/ellipse.png"))
print(pentagon("assets/pentagon.png"))
print(circle("assets/circle.png"))
print(rectangle("assets/rectangle.png"))
