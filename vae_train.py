import torch
import argparse
import os
import torch.nn.functional as F
import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np

from bbrl.models import VAE
from torch.optim import Adam
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter

def loss_fn(output, target):
    mse = F.mse_loss(output[0], target, reduction='sum')

    kl = -0.5 * torch.sum(1 + output[2] - output[1].pow(2) - output[2].exp())
    return mse + kl

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VAE on activations training and latent visualisation")
    parser.add_argument("command", metavar="<command>",
                        help="[train|visualise]")
    parser.add_argument("--dataset-path")
    parser.add_argument("--weights-path")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    from utils import ActivationsDataset
    # Set two classes for a train and test dataset

    vae = VAE()
    if args.command == "train":
        vae.to(device)
        writer = SummaryWriter("experiments/vae".format(args.epochs))
        acts_dataset_train = ActivationsDataset(args.dataset_path)
        acts_dataset_val = ActivationsDataset(args.dataset_path)
        indices = torch.randperm(len(acts_dataset_train)).tolist()
        train_index = round(len(indices) * 0.8)
        acts_dataset_train = torch.utils.data.Subset(acts_dataset_train, indices[:train_index])
        acts_dataset_val = torch.utils.data.Subset(acts_dataset_val, indices[train_index:])

        train_dataloader = torch.utils.data.DataLoader(acts_dataset_train, batch_size=128, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(acts_dataset_val, batch_size=128, shuffle=True)
        optimizer = Adam(vae.parameters(), lr=0.001)
        
        for i in range(args.epochs):
            print("Epoch: {}".format(i))
            total_loss = 0
            for data, targets, _ in train_dataloader:
                data = torch.stack(list(d.to(device) for d in data))
                targets = torch.stack(list(t.to(device) for t in targets))
                optimizer.zero_grad()
                outputs = vae(data)
                loss = loss_fn(outputs, targets)
                total_loss += loss.cpu().detach().item()
                loss.backward()
                optimizer.step()
            writer.add_scalar("train_loss", total_loss/len(train_dataloader), i)

            with torch.no_grad():
                total_eval_loss = 0
                for data, targets, _ in val_dataloader:
                    data = torch.stack(list(d.to(device) for d in data))
                    targets = torch.stack(list(t.to(device) for t in targets))
                    outputs = vae(data)
                    loss = loss_fn(outputs, targets)
                    total_eval_loss += loss.cpu().detach().item()
                writer.add_scalar("eval_loss", total_eval_loss/len(val_dataloader), i)
        torch.save(vae.state_dict(), args.weights_path)
    else:
        vae.to(device)
        vae.load_state_dict(torch.load(args.weights_path))
        vae.eval()
        acts_dataset_val = ActivationsDataset(args.dataset_path)
        indices = torch.randperm(len(acts_dataset_val)).tolist()
        train_index = round(len(indices) * 0.8)
        acts_dataset_val = torch.utils.data.Subset(acts_dataset_val, indices[train_index:])
        val_dataloader = torch.utils.data.DataLoader(acts_dataset_val, batch_size=128, shuffle=True)

        num_2_behaviour = ["approach", "grasp", "retract"]
        latent_acts = []
        labels = []
        for data, _, targets  in val_dataloader:
            data = torch.stack(list(d.to(device) for d in data))
            z_means, z_logvar = vae.encoder(data)
            latent_acts.extend(z_means.cpu().detach().numpy())
            labels.extend([num_2_behaviour[t] for t in targets.numpy()])

        latent_acts = np.array(latent_acts)
        labels = np.array(labels)
        tsne = TSNE(n_components=2, init="pca", random_state=0)
        # tsne = TSNE(n_components=3, init="pca", random_state=0)
        X = tsne.fit_transform(latent_acts)
        data = np.vstack((X.T, labels)).T
        df = pd.DataFrame(data=data, columns=["z1", "z2", "label"])
        # df = pd.DataFrame(data=data, columns=["z1", "z2", "z3", "label"])
        fig = px.scatter(df, x="z1", y="z2", color="label")
        # fig = px.scatter_3d(df, x="z1", y="z2", z="z3", color="label")
        fig.update_layout(
            title="VAE Latent Space",
            hovermode="x",
            title_x=0.5,
            font=dict(
                family="Courier New, monospace",
                size=18
            )
        )
        fig.show()
        # pio.write_html(fig, file="{}_3_comps.html".format(args.weights_path.split('.')[0]), auto_open=True)
