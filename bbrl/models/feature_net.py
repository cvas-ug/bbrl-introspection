import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight)

class FeatureNet(nn.Module):
    """
        Implements the feature extraction layers.
    """

    def __init__(self, internal_states=False):
        
        super(FeatureNet, self).__init__()
        # Feature extraction layers
        self.fc1 = nn.Linear(28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.internal_states = internal_states
        self.apply(init_weights)

    def forward(self, x):
        # Feedforward observation input to extract features
        x = F.elu(self.fc1(x))
        output = F.elu(self.fc2(x))

        if self.internal_states:
            return x, output
        return output