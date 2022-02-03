import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from mnist_dataset import MNISTSummation
from models import InvariantModelNoEmb, SmallMNISTCNN




def train_1_epoch(model, train_db, optimizer, epoch_num: int = 0):
    model.train()

    epoch_loss = []
    for i in tqdm(range(len(train_db))):
        loss = train_1_item(model, train_db, optimizer, i)
        epoch_loss.append(loss)
    
    print(f'{epoch_num}: train_loss {np.sum(epoch_loss)/ len(train_db)}' )

def train_1_item(model, train_db, optimizer, item_number: int) -> float:
    x, y_list, target = train_db.__getitem__(item_number)
    if torch.cuda.is_available():
        x, target = x.cuda(), target.cuda()

    # x, target = Variable(x), Variable(target)

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
    # totals = [0] * 51
    # corrects = [0] * 51

    # for i in tqdm(range(len(test_db))):
    #     x, y_list, target = test_db.__getitem__(i)

    #     item_size = x.shape[0]

    #     if torch.cuda.is_available():
    #         x = x.cuda()

    #     pred = model.forward(Variable(x)).data



    #     if torch.cuda.is_available():
    #         pred = pred.cpu().numpy().flatten()

    #     pred = int(round(float(pred[0])))
    #     target = int(round(float(target.numpy()[0])))

    #     totals[item_size] += 1

    #     if pred == target:
    #         corrects[item_size] += 1


    x, y_list, target = test_db.__getitem__(0)
    if torch.cuda.is_available():
            x = x.cuda()
    predd = model.phi.forward(x)
    print(f"Original: {y_list}")
    print(f"Predicted: {predd.view(-1).data}")
    
    # totals = np.array(totals)
    # corrects = np.array(corrects)

    # print(corrects / totals)



lr = 1e-3
wd = 5e-3
train_db = MNISTSummation(min_len=2, max_len=10, dataset_len=10000, train=True)
test_db = MNISTSummation(min_len=5, max_len=50, dataset_len=1000, train=False)

the_phi = SmallMNISTCNN()

model = InvariantModelNoEmb(phi=the_phi)
if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


for epoch in range(10):
    train_1_epoch(model, train_db, optimizer, epoch)
    evaluate(model, test_db)