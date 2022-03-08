import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from french_setyield_dataloader import FrenchLSTMSetYieldDataset
from french_ae_dataloader import FrenchSimHistDataset
from french_yield_models import YieldRNNModel
from utils.constants import DEPTS, YEARS
from utils.simiarlity_utils import wasserstein_severity


from torch.utils.data import DataLoader
import glob

def train_1_epoch(model, train_db, optimizer, epoch_num: int = 0):
    model.train()

    epoch_loss = []
    for i in tqdm(range(len(train_db))):
        loss = train_1_item(model, train_db, optimizer, i)
        epoch_loss.append(loss)
    
    print(f'{epoch_num}: train_loss {np.sum(epoch_loss)/ len(train_db)}' )



def train_sim_1_epoch(model, oracle_dl, optimizer, epoch_num: int = 0):
    model.train()
    epoch_loss = []
    for x, y, indicator in tqdm(oracle_dl):
        x, y, indicator = x.cuda(), y.cuda(), indicator.cuda()

        # print(indicator)

        optimizer.zero_grad()
        model_x = model(x, torch.tensor([6]*x.shape[0]))
        model_y = model(y, torch.tensor([6]*x.shape[0]))

        # loss = F.mse_loss(model_x, model_y, reduction="none")
        loss = F.l1_loss(model_x, model_y, reduction="none")

        loss = indicator * loss.view(-1)
        loss = loss.mean()

        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

    print(f"{epoch_num}: oracle train_loss {np.sum(epoch_loss)/ (len(oracle_dl) * 5)}")



def train_1_item(model, train_db, optimizer, item_number: int) -> float:
    x, target = train_db.__getitem__(item_number)
    if torch.cuda.is_available():
        x, target = x.cuda(), target.cuda()
    
    # print(x.shape)

    optimizer.zero_grad()
    pred = model.forward(x, torch.tensor([6]*x.shape[0]))
    # the_loss = F.mse_loss(pred, target)
    the_loss = F.l1_loss(pred, target)
    

    the_loss.backward()
    optimizer.step()

    the_loss_tensor = the_loss.data
    if torch.cuda.is_available():
        the_loss_tensor = the_loss_tensor.cpu()

    the_loss_numpy = the_loss_tensor.numpy().flatten()
    the_loss_float = float(the_loss_numpy[0])

    if item_number % 100 == 0:
        print(f'target: {target} pred: {pred} diff {target - pred}')

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

width_dim = 32


train_db = FrenchLSTMSetYieldDataset("./winter_wheat_filtered_2002.csv", "../french_dept_data", width = width_dim)





train_subtile_paths = []
for dept in DEPTS:
    for year in YEARS:
        train_subtile_paths.extend(glob.glob(f"../french_dept_data/{dept}/{year}/split*_{width_dim}/*"))

sim_dataset =  FrenchSimHistDataset(train_subtile_paths, normalize= False, width = width_dim, similarity_func=wasserstein_severity)
sim_dataloader = DataLoader(sim_dataset, batch_size=32)
model = YieldRNNModel(input_dim=32*32, bidirectional=False)

if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


for epoch in range(50):
    # evaluate(model, test_db)


    # train_1_epoch(model, train_db, optimizer, epoch)

    # if epoch > 5:
    train_sim_1_epoch(model, sim_dataloader, optimizer, epoch)

    train_1_epoch(model, train_db, optimizer, epoch)
    # evaluate(model, test_db)