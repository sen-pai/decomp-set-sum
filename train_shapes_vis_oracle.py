import numpy as np

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from shapes_dataset import ShapesSummation, VisualSimOracle
from models import InvariantModelNoEmb, SmallShapesCNN


def train_1_epoch(model, train_db, optimizer, epoch_num: int = 0):
    model.train()

    epoch_loss = []
    for i in tqdm(range(len(train_db))):
        loss = train_1_item(model, train_db, optimizer, i)
        epoch_loss.append(loss)

    print(f"{epoch_num}: train_loss {np.sum(epoch_loss)/ len(train_db)}")


def train_vis_sim_oracle_1_epoch(model, oracle_dl, optimizer, epoch_num: int = 0):
    model.train()
    epoch_loss = []
    for x, y, indicator in tqdm(oracle_dl):
        x, y, indicator = x.cuda(), y.cuda(), indicator.cuda()

        optimizer.zero_grad()
        model_x = model.phi(x)
        model_y = model.phi(y)

        # loss = F.mse_loss(model_x, model_y, reduction="none")
        loss = F.l1_loss(model_x, model_y, reduction="none")
        
        loss = indicator * loss.view(-1)
        loss = loss.mean()

        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

    print(f"{epoch_num}: oracle train_loss {np.sum(epoch_loss)/ (len(oracle_dl) * 5)}")


def train_1_item(model, train_db, optimizer, item_number: int) -> float:
    x, _, target = train_db.__getitem__(item_number)
    if torch.cuda.is_available():
        x, target = x.cuda(), target.cuda()

    optimizer.zero_grad()
    pred = model.forward(x)
    the_loss = F.mse_loss(pred, target)

    the_loss.backward()
    optimizer.step()

    the_loss_tensor = the_loss.data
    if torch.cuda.is_available():
        the_loss_tensor = the_loss_tensor.cpu()

    the_loss_numpy = the_loss_tensor.numpy().flatten()
    the_loss_float = float(the_loss_numpy[0])

    return the_loss_float


def evaluate(model, test_db):
    model.eval()

    x, y_list, target = test_db.__getitem__(0)
    if torch.cuda.is_available():
        x = x.cuda()
    predd = model.phi.forward(x)
    print(f"Original: {y_list}")
    print(f"Predicted: {predd.view(-1).data}")


lr = 1e-3
wd = 5e-3

train_db = ShapesSummation(min_len=2, max_len=5, dataset_len=10000)
test_db = ShapesSummation(min_len=5, max_len=20, dataset_len=10)

oracle = VisualSimOracle(100)
oracle_dataloader = DataLoader(oracle, batch_size=5)

the_phi = SmallShapesCNN()

model = InvariantModelNoEmb(phi=the_phi)
if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


for epoch in range(50):
    train_1_epoch(model, train_db, optimizer, epoch)

    # if epoch > 5:
    train_vis_sim_oracle_1_epoch(model, oracle_dataloader, optimizer, epoch)

    evaluate(model, test_db)