import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from french_setyield_dataloader import FrenchPickleDataset
from french_yield_models import YieldGroupsRNNModel


def train_1_epoch(model, train_db, optimizer, epoch_num: int = 0):
    model.train()

    epoch_loss = []
    for i in tqdm(range(len(train_db))):
        loss = train_1_item(model, train_db, optimizer, i)
        epoch_loss.append(loss)

    print(f"{epoch_num}: train_loss {np.sum(epoch_loss)/ len(train_db)}")


def train_1_item(model, train_db, optimizer, item_number: int) -> float:
    x, group_sizes, target = train_db.__getitem__(item_number)
    if torch.cuda.is_available():
        x, group_sizes, target = x.cuda(), group_sizes.cuda(), target.cuda()

    # print(target)
    # print(x.shape)

    optimizer.zero_grad()
    pred = model.forward(x, torch.tensor([6] * x.shape[0]), group_sizes)
    the_loss = F.mse_loss(pred, target)
    # the_loss = F.l1_loss(pred, target)

    the_loss.backward()
    optimizer.step()

    the_loss_tensor = the_loss.data
    if torch.cuda.is_available():
        the_loss_tensor = the_loss_tensor.cpu()

    the_loss_numpy = the_loss_tensor.numpy().flatten()
    the_loss_float = float(the_loss_numpy[0])
    # print(pred)
    if item_number % 50 == 0:
        print(f"target: {target} pred: {pred} diff {target - pred}")

    return the_loss_float


# def evaluate(model, test_db):
#     model.eval()

#     x, y_list, target = test_db.__getitem__(0)
#     if torch.cuda.is_available():
#             x = x.cuda()
#     predd = model.phi.forward(x)
#     print(f"Original: {y_list}")
#     print(f"Predicted: {predd.view(-1).data}")


lr = 1e-3
wd = 5e-3
train_db = FrenchPickleDataset(
    "./winter_wheat_filtered_2002.csv",
    "../french_dept_data",
    [
        "Alpes_de_Haute_Provence",
        "Alpes_Maritimes",
        "Ardeche",
        "Ariege",
        "Aude",
        "Aveyron",
        "Bouches_du_Rhone",
        "Cantal",
        "Correze",
        "Corse_du_Sud",
        "Creuse",
        "Dordogne",
        "Doubs",
        "Drome",
        "Gard",
        "Gironde",
        "Haut_Rhin",
        "Haute_Corse",
        "Haute_Loire",
        "Haute_Savoie",
        "Haute_Vienne",
        "Hautes_Alpes",
        "Hautes_pyrenees",
        "Herault",
        "Jura",
        "Landes",
        "Loire",
        "Lot",
        "Lozere",
        "Pyrenees_Atlantiques",
        "Pyrenees_Orientales",
        "Rhone",
        "Savoie",
        "Seine_Saint_Denis",
        "Tarn_et_Garonne",
        "Territoire_de_Belfort",
        "Val_de_Marne",
        "Var",
        "Vaucluse",
        "Vosges",
    ],
    norm_sum=10000,
)
#  ['Ain', 'Aisne', 'Allier' , 'Ardennes', 'Ariege', 'Aube']

model = YieldGroupsRNNModel(input_dim=1, bidirectional=False)

if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


for epoch in range(50):
    train_1_epoch(model, train_db, optimizer, epoch)
    # evaluate(model, test_db)
