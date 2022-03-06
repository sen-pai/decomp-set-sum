
import os
import numpy as np
import random
import glob
import copy
import shutil
from collections import defaultdict


import argparse
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from french_ae_dataloader import FrenchAEDataset
from ae_model import YieldVAE

from utils.constants import TrainDEPTS, TrainYEARS, ValDEPTS, ValYEARS

parser = argparse.ArgumentParser(description="AE")



parser.add_argument(
    "--exp-name",
    default="ae_picardie",
    help="Experiment name",
)
parser.add_argument(
    "--save-freq",
    type=int,
    default=10,
    help="every x epochs save model weights",
)
parser.add_argument(
    "--plot-freq",
    type=int,
    default=1,
    help="every x epochs plot predictions",
)
parser.add_argument(
    "--batch",
    type=int,
    default=64,
    help="train batch size",
)
parser.add_argument(
    "--vbatch",
    type=int,
    default=64,
    help="validation batch size",
)
parser.add_argument(
    "--vs",
    type=float,
    default=0.05,
    help="val split",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.0001,
    help="initial lr",
)

parser.add_argument(
    "--schedule",
    action="store_true",
    help="lr scheduler",
)

args = parser.parse_args()

# fix random seeds
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)



current_dir = os.getcwd()

train_subtile_paths = []
for dept in TrainDEPTS:
    for year in TrainYEARS:
        train_subtile_paths.extend(glob.glob(f"../french_dept_data/{dept}/{year}/split*_64/*"))

train_dataset = FrenchAEDataset(train_subtile_paths, normalize= True)



val_subtile_paths = []
for dept in ValDEPTS:
    for year in ValYEARS:
        val_subtile_paths.extend(glob.glob(f"../french_dept_data/{dept}/{year}/split*_64/*"))

valid_dataset = FrenchAEDataset(val_subtile_paths, normalize= True)

dataloaders = {
    "train": DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    ),
    "val": DataLoader(
        valid_dataset, batch_size=args.vbatch, shuffle=False, num_workers=0
    ),
}


# experiments_path = os.path.join(current_dir, "experiments")
# exp_name_path = os.path.join(experiments_path, exp_name)
# os.chdir(experiments_path)

# if os.path.exists(exp_name):
#     shutil.rmtree(exp_name)

# os.mkdir(exp_name)
# os.chdir(exp_name)
# os.mkdir("weights")
# os.chdir(current_dir)


def calc_loss(
    prediction,
    target,
    metrics,
):
    mse_loss = F.mse_loss(prediction, target)
    mae_loss = F.l1_loss(prediction, target)
    metrics["MSE"] += mse_loss.data.cpu().numpy() * target.size(0)
    metrics["MAE"] += mae_loss.data.cpu().numpy() * target.size(0)

    return mse_loss



# def calc_loss(recon_x, x, mu, logvar, metrics):
#     # BCE = F.binary_cross_entropy(recon_x, x)
#     BCE = F.mse_loss(recon_x, x)

#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

#     metrics["BCE"] += BCE.data.cpu().numpy() * recon_x.size(0)
#     metrics["KLD"] += KLD.data.cpu().numpy() * recon_x.size(0)

#     return BCE + KLD

def print_metrics(metrics, epoch_samples, phase, epoch):
    outputs = []
    outputs.append("{}:".format(str(epoch)))
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, num_epochs=151):
    # model.load_state_dict(torch.load("weights.pt"))
    # print("loaded weights")
    begin = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)

            epoch_samples = 0

            count = 0
            for input_targets, targets in tqdm(dataloaders[phase]):
                count += 1

                if count == 100:
                    print(targets[0][0][0])
                # if phase == "train":
                input_targets, targets, = (
                    input_targets.to(device),
                    targets.to(device),
                )
                # else:
                #     input_targets = input_targets.to(device)

                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    optimizer.zero_grad()
                    predictions, mu, var = model(input_targets)
                    predictions, mu, var = predictions.to(device), mu.to(device), var.to(device)
                    if count == 100:
                        print(predictions[0][0][0])
                    # if phase == "val":
                    #     predictions = predictions.to("cpu")

                    loss = calc_loss(
                        predictions,
                        targets,
                        metrics,
                    )

                    # loss = calc_loss(
                    #     predictions,
                    #     targets,
                    #     mu,
                    #     var,
                    #     metrics,
                    # )

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += input_targets.size(0)

            print_metrics(metrics, epoch_samples, phase, epoch)
            check_lr = optimizer.param_groups[0]["lr"]
            print("LR", check_lr)

            if args.schedule:
                if check_lr != min_lr:
                    scheduler.step()

            end = time.time()

            # deep copy the model
            # if phase == "val" and epoch % args.save_freq == 0:
            #     print("saving model")
            #     best_model_wts = copy.deepcopy(model.state_dict())
            #     os.chdir(os.path.join(exp_name_path, "weights"))
            #     weight_name = args.exp_name + "_weights_" + str(epoch) + ".pt"
            #     torch.save(best_model_wts, weight_name)
            #     os.chdir(current_dir)

device = torch.device('cuda')

model = YieldVAE().to(device)

# all parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = None
if args.schedule:
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    global min_lr
    min_lr = 0.0001

train_model(model, optimizer, scheduler, num_epochs=601)