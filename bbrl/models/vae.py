import torch
import torch.nn as nn

import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(256, 400)
        self.fc2 = nn.Linear(400, 128)

        self.z_mean = nn.Linear(128, 50)
        self.z_scale = nn.Linear(128, 50)
    
    def forward(self, x):

        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)

        z_mu = self.z_mean(x)
        z_logvar = self.z_scale(x)

        return z_mu, z_logvar

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(50, 128)
        self.fc2 = nn.Linear(128, 400)
        self.fc3 = nn.Linear(400, 256)
    
    def forward(self, x):

        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        output = F.elu(x)

        return output