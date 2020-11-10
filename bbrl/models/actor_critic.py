import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight)

class ActorCritic(nn.Module):
    """
        Implements the A3C choreographer.
    """
    def __init__(self, internal_states):
        super(ActorCritic, self).__init__()
        # Naive way to change dim input to LSTM if training
        # with internal statess
        # 1) Original AC -> 128
        # 2) Latent Means Concat to FE -> 178
        # 3) Latent Means Only -> 50
        # 4) Latent Means & Log Variance -> 100
        # 5) Sampling from Latent Space -> 50
        self.input_dim = 50 if internal_states else 128

        self.lstm = nn.LSTMCell(self.input_dim, 32)
        self.critic_linear = nn.Linear(32, 1)
        self.actor_linear = nn.Linear(32, 3)

        self.apply(init_weights)

    def forward(self, x, hx, cx):
        # Feedforward features to get behaviour and state value.
        lstm_input = x.view(-1, self.input_dim)
        hx, cx = self.lstm(lstm_input, (hx, cx))
        state_value = self.critic_linear(hx)
        actions = self.actor_linear(hx)

        output = {}
        output["state"] = state_value
        output["actions"] = actions
        output["hidden"] = hx
        output["cell"] = cx
    
        return output