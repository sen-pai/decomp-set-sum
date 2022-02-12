import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from vegetation_dataset import VegetationDryness
from veg_models import InvariantModelNoEmb, MlpPhi


def train_1_epoch(model, train_db, optimizer, epoch_num: int = 0):
    model.train()

    epoch_loss = []
    for i in tqdm(range(len(train_db))):
        loss = train_1_item(model, train_db, optimizer, i)
        epoch_loss.append(loss)

    print(f"{epoch_num}: train_loss {np.sum(epoch_loss)/ len(train_db)}")


def train_1_item(model, train_db, optimizer, item_number: int) -> float:
    x, _, target = train_db.__getitem__(item_number)

    if torch.cuda.is_available():
        x, target = x.cuda(), target.cuda()

    optimizer.zero_grad()
    pred = model.forward(x)
    # the_loss = F.mse_loss(pred, target, reduction='mean')
    the_loss = F.l1_loss(pred, target)

    # print(the_loss, pred, target)

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
    print(f"Predicted: {predd.view(-1).data }")


def full_eval_loss(model, full_test_db):
    model.eval()

    losses = []
    count = 0
    for x, _, t in full_test_db:
        count += 1
        if count > 800:
            print(f"Eval MSE: {np.sum(np.sqrt(losses))/len(losses)}")
            break
        if torch.cuda.is_available():
            x = x.cuda()
            t = t.cuda()
        pred = model.phi.forward(x)
        losses.append(F.mse_loss(pred, t).item())


lr = 1e-3
wd = 5e-3
std = 39
mean = 104
train_db = VegetationDryness(
    min_len=2, max_len=5, dataset_path="input_data.csv", norm=False, split="sy"
)
test_db = VegetationDryness(
    min_len=5,
    max_len=20,
    dataset_path="input_data.csv",
    is_train=False,
    norm=False,
    split="sy",
)
full_test_db = VegetationDryness(
    min_len=1,
    max_len=1,
    dataset_path="input_data.csv",
    is_train=False,
    norm=False,
    split="sy",
)


the_phi = MlpPhi()

model = InvariantModelNoEmb(phi=the_phi)
if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


for epoch in range(50):
    train_1_epoch(model, train_db, optimizer, epoch)
    evaluate(model, test_db)
    full_eval_loss(model, full_test_db)