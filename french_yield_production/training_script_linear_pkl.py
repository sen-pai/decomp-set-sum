import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from french_setyield_dataloader import FrenchLinearPickleDataset, FrenchSimLinearPickleDataset
from french_yield_models import YieldLinearModel

from torch.utils.data import DataLoader


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
    pred = model.forward(x, group_sizes)
    the_loss = F.mse_loss(pred, target)
    # the_loss = F.l1_loss(pred, target)

    the_loss.backward()
    optimizer.step()

    the_loss_tensor = the_loss.data
    if torch.cuda.is_available():
        the_loss_tensor = the_loss_tensor.cpu()

    the_loss_numpy = the_loss_tensor.numpy().flatten()
    the_loss_float = float(the_loss_numpy[0])

    # print(the_loss_float)
    # print(pred)
    if item_number % 10 == 0:
        print(f"target: {target} pred: {pred} diff {target - pred}, {the_loss_float}")

    return the_loss_float




def train_sim_1_epoch(model, oracle_dl, optimizer, epoch_num: int = 0):
    model.train()
    epoch_loss = []
    for x, y, indicator in tqdm(oracle_dl):
        # indicator = indicator.cuda()
        x, y, indicator = x.cuda(), y.cuda(), indicator.cuda()

        # print(x.shape)
        # print(indicator)

        optimizer.zero_grad()
        model_x = model.encoder(x)
        model_y = model.encoder(y)

        # print(model_x, model_y, indicator)

        loss = F.mse_loss(model_x, model_y, reduction="none")
        # loss = F.l1_loss(model_x, model_y, reduction="none")

        loss = indicator * loss.view(-1)
        loss = loss.mean()

        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

        # print(loss.item())

    print(f"{epoch_num}: oracle train_loss {np.sum(epoch_loss)/ (len(oracle_dl))}")




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


areas = [
        'Ain', 'Aisne', 'Allier', 'Ardennes', 'Aube', 'Bas_Rhin',
       'Calvados', 'Charente', 'Charente_Maritime', 'Cher', 'Cote_d_Or',
    #    'Cotes_d_Armor', 'Deux_Sevres', 'Dordogne', 'Drome', 'Essonne',
    #    'Eure', 'Eure_et_Loir', 'Finistere', 'Gers', 'Haut_Rhin',
    #    'Haute_Garonne', 'Haute_Marne', 'Haute_Saone', 'Ille_et_Vilaine',
    #    'Indre', 'Indre_et_Loire', 'Isere', 'Jura', 'Loir_et_Cher',
    #    'Loire_Atlantique', 'Loiret', 'Lot_et_Garonne', 'Maine_et_Loire',
    #    'Manche', 'Marne', 'Mayenne', 'Meurthe_et_Moselle', 'Meuse',
    #    'Morbihan', 'Moselle', 'Nievre', 'Nord', 'Oise', 'Orne',
    #    'Pas_de_Calais', 'Puy_de_Dome', 'Saone_et_Loire', 'Sarthe',
    #    'Seine_et_Marne', 'Seine_Maritime', 'Somme', 'Tarn',
    #    'Tarn_et_Garonne', 'Val_d_Oise', 'Vendee', 'Vienne', 'Vosges',
    #    'Yonne', 'Yvelines'
    ]
train_db = FrenchLinearPickleDataset(
    "./winter_wheat_filtered_2002.csv",
    "../french_dept_data",
    areas,
    norm_sum=10000,
)
#  ['Ain', 'Aisne', 'Allier' , 'Ardennes', 'Ariege', 'Aube']


sim_dataset = FrenchSimLinearPickleDataset("./winter_wheat_filtered_2002.csv", "../french_dept_data", areas, norm_sum = 10000)

sim_dataloader = DataLoader(sim_dataset, batch_size=32)


model = YieldLinearModel()

if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


for epoch in range(1000):

    train_sim_1_epoch(model, sim_dataloader, optimizer, epoch)
    train_1_epoch(model, train_db, optimizer, epoch)
    # evaluate(model, test_db)
