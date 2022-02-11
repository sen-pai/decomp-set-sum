from typing import Tuple

import numpy as np
import pandas as pd
import random

import torch
from torch import FloatTensor
from torch.utils.data.dataset import Dataset

import copy


def cleanup_df(df: pd.DataFrame) -> pd.DataFrame:
    # print(df.columns)
    del df["Unnamed: 0"]
    df["date"] = pd.to_datetime(df["date"]).dt.year

    return df


def train_test_split(
    df: pd.DataFrame, type: str = "year"
) -> Tuple[np.array, np.array, np.array, np.array]:
    cleaned_df = cleanup_df(df)

    if type == "year":
        del cleaned_df["site"]

        train = cleaned_df.loc[(cleaned_df.date < 2018)]
        test = cleaned_df.loc[(cleaned_df.date >= 2018)]

        train_y = train["percent(t)"].to_numpy()
        test_y = test["percent(t)"].to_numpy()

        train.drop(["date", "percent(t)"], axis=1, inplace=True)
        test.drop(["date", "percent(t)"], axis=1, inplace=True)

        return train.to_numpy(), train_y, test.to_numpy(), test_y


class VegetationDryness(Dataset):
    def __init__(
        self,
        min_len: int,
        max_len: int,
        dataset_path: str,
        split: str = "year",
        is_train: bool = True,
        norm: bool = True
    ):
        self.min_len = min_len
        self.max_len = max_len
        self.dataset = pd.read_csv(dataset_path)
        self.mean = 104
        self.std = 39
        self.norm = norm

        if is_train:
            self.elements, self.percents, _, _ = train_test_split(self.dataset, split)
        else:
            _, _, self.elements, self.percents = train_test_split(self.dataset, split)

        self.items_len_range = list(range(self.min_len, self.max_len + 1))

    def __len__(self) -> int:
        return self.elements.shape[0]

    def __getitem__(self, item: int) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:

        set_size = random.choice(self.items_len_range)
        elements_idx = np.random.randint(self.elements.shape[0], size= set_size)
        set_elements = self.elements[elements_idx, :]

        set_percents = self.percents[elements_idx]
        if self.norm:
            set_percents = (set_percents - self.mean)/ self.std
            
        the_sum = np.sum(set_percents)

        return (
            torch.tensor(set_elements).float(),
            torch.tensor(set_percents),
            torch.tensor(the_sum).float(),
        )


# class VisualSimOracle(Dataset):
#     def __init__(
#         self,
#         dataset_len: int,
#         same_prob:float = 0.5
#     ):
#         """
#         Return a tuple of two random (x, y) and 1 if x = y, or -1 if x != y
#         """
#         self.dataset_len = dataset_len
#         self.same_prob = same_prob
#         self.shape_funcs = [ellipse, circle, pentagon, rectangle]

#     def __len__(self) -> int:
#         return self.dataset_len

#     def torchify(self, x: np.array, norm: bool = True) -> torch.tensor:
#         x = torch.tensor(x)
#         if norm:
#             return x / 255.0
#         return x

#     def __getitem__(self, item: int) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:

#         x_img, x_target = self.shape_funcs[random.randint(0, 3)]()
#         x_img = self.torchify(x_img)

#         if random.uniform(0, 1) >= self.same_prob:
#             y_img, y_target = self.shape_funcs[random.randint(0, 3)]()
#             y_img = self.torchify(y_img)
#         else:
#             y_img = copy.deepcopy(x_img)
#             y_target = copy.deepcopy(x_target)

#         indicator = 1 if x_target == y_target else -1

#         return x_img, y_img, indicator


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    shapes = VegetationDryness(2, 5, 'input_data.csv')
    stack, tar, su = shapes.__getitem__(0)
    print(stack.shape)
    print(tar)
    print(su)

    # oracle = VisualSimOracle(100)
    # x, y, i = oracle.__getitem__(0)
    # # print(x.shape, i)

    # oracle_dataloader = DataLoader(oracle, batch_size=5)
    # b_x, b_y, b_i = next(iter(oracle_dataloader))
    # print(b_i)
