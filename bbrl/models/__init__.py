import os
import torch
import torch.nn as nn

from .feature_net import FeatureNet
from .reactive_net import ReactiveNet
from .actor_critic import ActorCritic
from .vae import Encoder, Decoder

LOW_LEVEL_BEHAVIOURS = [ "approach", "grasp", "retract" ]

class BehaviourNetwork(nn.Module):
    """
        Implements the path from feature extraction to reactive network output
    """
    def __init__(self, weights_path, behaviour=None):
        super(BehaviourNetwork, self).__init__()
        self.feature_net = FeatureNet()
        self.reactive_net = ReactiveNet()
        self.behaviour = behaviour
        self.weights_path = weights_path if behaviour in LOW_LEVEL_BEHAVIOURS else os.path.dirname(os.path.dirname(weights_path))

        # Load existing weights
        self.load_weights()

        # Freeze weights here
        if self.behaviour in LOW_LEVEL_BEHAVIOURS:
            self.freeze_weights()
    
    def forward(self, x):
        # Feedforward input to behaviour network path
        features = self.feature_net(x)
        output = self.reactive_net(features)
        
        return features, output

    def freeze_weights(self):
        # Freezes weights except the ones needed to be trained
        for name, param in self.reactive_net.named_parameters():
            layer_name = name.split('.')[0]
            if layer_name not in self.reactive_net.get_trainable_layers(self.behaviour):
                param.requires_grad = False

        if self.behaviour != "approach":
            for param in self.feature_net.parameters():
                param.requires_grad = False

    def load_weights(self):
        # Load weights if behaviour is other than approach
        if self.behaviour != "approach":
            self.feature_net.load_state_dict(torch.load(os.path.join(self.weights_path, "feature_net.pth")))
            self.reactive_net.load_state_dict(torch.load(os.path.join(self.weights_path, "reactive_net.pth")))

    def save_model_weights(self):
        # Save weights of feature and reactive networks independently
        if self.behaviour == "approach":
            torch.save(self.feature_net.state_dict(), os.path.join(self.weights_path, "feature_net.pth"))
        torch.save(self.reactive_net.state_dict(), os.path.join(self.weights_path, "reactive_net.pth"))

class ChoreographNetwork(nn.Module):
    """
        Implements the path from feature extraction to A3C output
    """
    def __init__(self, weights_path, internal_states=False):
        super(ChoreographNetwork, self).__init__()
        self.feature_net = FeatureNet(internal_states)
        self.ac = ActorCritic(internal_states)
        self.weights_path = weights_path
        self.internal_states = internal_states
        if self.internal_states:
            self.vae = VAE()
            self.vae.load_state_dict(torch.load(os.path.join(os.path.join(os.path.dirname(os.path.dirname(self.weights_path)), "vae"), "vae.pth")))
            self.vae.eval()
        # Load existing weights
        self.load_weights()

        # Freeze weights here
        self.freeze_weights()
    
    def forward(self, x, hx, cx):
        # Feedforward input to feature and choreographer network path
        ac_input_features = self.feature_net(x)
        if self.internal_states:
            first_layer_features, second_layer_features = ac_input_features
            vae_input = torch.cat((first_layer_features, second_layer_features))
            # ac_input_features, _ = self.vae.encoder(vae_input) # only internal states
            z_means, z_logvar = self.vae.encoder(vae_input)
            std = torch.exp(0.5*z_logvar)
            eps = torch.randn_like(std)
            ac_input_features = z_means+eps*std
            # ac_input_features = torch.cat((second_layer_features, z_means)) # concatenating internal states
            # ac_input_features = torch.cat((z_means, z_logvar))
        output = self.ac(ac_input_features, hx, cx)

        return output

    def load_weights(self):
        self.feature_net.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.dirname(self.weights_path)), "feature_net.pth")))
        if (os.path.exists(os.path.join(self.weights_path, "ac.pth"))):
            self.ac.load_state_dict(torch.load(os.path.join(self.weights_path, "ac.pth")))

    def freeze_weights(self):
        # Freezes weights of the Feature Network
        for param in self.feature_net.parameters():
            param.requires_grad = False

    def save_model_weights(self):
        # Save weights of feature and reactive networks independently
        torch.save(self.ac.state_dict(), os.path.join(self.weights_path, "ac_10.pth"))

class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):

        z_mean, z_logvar = self.encoder(x)
        std = torch.exp(0.5*z_logvar)
        eps = torch.randn_like(std)
        output = self.decoder(z_mean+eps*std)

        return output, z_mean, z_logvar