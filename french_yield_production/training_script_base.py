import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from french_setyield_dataloader import FrenchSetYieldDataset
from french_yield_models import YieldModel


def train_1_epoch(model, train_db, optimizer, epoch_num: int = 0):
    model.train()

    epoch_loss = []
    for i in tqdm(range(len(train_db))):
        loss = train_1_item(model, train_db, optimizer, i)
        epoch_loss.append(loss)
    
    print(f'{epoch_num}: train_loss {np.sum(epoch_loss)/ len(train_db)}' )

def train_1_item(model, train_db, optimizer, item_number: int) -> float:
    x, target = train_db.__getitem__(item_number)
    if torch.cuda.is_available():
        x, target = x.cuda(), target.cuda()


    optimizer.zero_grad()
    pred = model.forward(x)
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
train_db = FrenchSetYieldDataset("./winter_wheat_filtered_2002.csv", "../french_dept_data")


model = YieldModel()

if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


for epoch in range(50):
    train_1_epoch(model, train_db, optimizer, epoch)
    # evaluate(model, test_db)