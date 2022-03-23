import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from french_setyield_dataloader import FrenchLinearPickleDataset
from french_yield_models import YieldLinearModel


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
train_db = FrenchLinearPickleDataset(
    "./winter_wheat_filtered_2002.csv",
    "../french_dept_data",
    [
        'Ain', 'Aisne', 'Allier', 'Ardennes', 'Aube', 'Bas_Rhin',
       'Calvados', 'Charente', 'Charente_Maritime', 'Cher', 'Cote_d_Or',
       'Cotes_d_Armor', 'Deux_Sevres', 'Dordogne', 'Drome', 'Essonne',
       'Eure', 'Eure_et_Loir', 'Finistere', 'Gers', 'Haut_Rhin',
       'Haute_Garonne', 'Haute_Marne', 'Haute_Saone', 'Ille_et_Vilaine',
       'Indre', 'Indre_et_Loire', 'Isere', 'Jura', 'Loir_et_Cher',
       'Loire_Atlantique', 'Loiret', 'Lot_et_Garonne', 'Maine_et_Loire',
       'Manche', 'Marne', 'Mayenne', 'Meurthe_et_Moselle', 'Meuse',
       'Morbihan', 'Moselle', 'Nievre', 'Nord', 'Oise', 'Orne',
       'Pas_de_Calais', 'Puy_de_Dome', 'Saone_et_Loire', 'Sarthe',
       'Seine_et_Marne', 'Seine_Maritime', 'Somme', 'Tarn',
       'Tarn_et_Garonne', 'Val_d_Oise', 'Vendee', 'Vienne', 'Vosges',
       'Yonne', 'Yvelines'
    ],
    norm_sum=10000,
)
#  ['Ain', 'Aisne', 'Allier' , 'Ardennes', 'Ariege', 'Aube']

model = YieldLinearModel()

if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


for epoch in range(50):
    train_1_epoch(model, train_db, optimizer, epoch)
    # evaluate(model, test_db)
