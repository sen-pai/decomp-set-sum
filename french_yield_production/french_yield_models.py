import torch
import torch.nn as nn



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class YieldModel(nn.Module):
    def __init__(self, image_channels=6, h_dim=1024, z_dim=32):
        super(YieldModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(h_dim, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        pred = self.encoder(x)
        return torch.sum(pred)



if __name__ == "__main__":
    k = torch.rand(2, 6, 64, 64)
    print(k.shape)

    vae = YieldModel()

    print(vae(k))