import os
import plotly
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--path")
    parser.add_argument("--noise", action="store_true")
    args = parser.parse_args()
    
    if args.command == "value":
        colours = ["blue", "red", "green"]
        data_files = next(os.walk(args.path))[2]
        data_files = sorted(data_files)
        behaviours = ["approach value", "grasp value", "retract value"]
        behaviour_index = 0
        step = 10
        for i in range(0, len(data_files), step):
            behaviour_name = behaviours[behaviour_index]
            behaviour_data_files = data_files[i:i+step]
            dfs = []
            read_first = False
            cols = ["Step", "Value"]
            for j, behaviour_data_f in enumerate(behaviour_data_files):
                if read_first:
                    cols = ["Value"]
                df = pd.read_csv(os.path.join(args.path, behaviour_data_f), usecols=cols)
                df.rename(columns={"Value": behaviour_name + "_{}".format(j)}, inplace=True)
                dfs.append(df)
                read_first = True
            df = pd.concat(dfs, axis=1)
            
            df["mean"] = df.loc[:, behaviour_name + "_0" : behaviour_name + "_{}".format(step - 1)].mean(axis=1)
            df["std"] = df.loc[:, behaviour_name + "_0" : behaviour_name + "_{}".format(step - 1)].std(axis=1).round(2)
            df["mean"].to_csv(f"means_logvar_{behaviour_name}.csv", index=False)
            behaviour_index += 1
        # fig = go.Figure([
        #     go.Scatter(
        #         name="State Value",
        #         x=df["Step"],
        #         y=df["mean"],
        #         mode="lines",
        #         marker=dict(color=colours[0])
        #     ),
        #     go.Scatter(
        #         x=df["Step"],
        #         y=df["mean"]+df["std"],
        #         mode="lines",
        #         marker=dict(color="#444"),
        #         line=dict(width=0),
        #         showlegend=False
        #     ),
        #     go.Scatter(
        #         x=df["Step"],
        #         y=df["mean"]-df["std"],
        #         marker=dict(color="#444"),
        #         line=dict(width=0),
        #         mode="lines",
        #         fillcolor='rgba(68, 68, 68, 0.3)',
        #         fill='tonexty',
        #         showlegend=False
        #     )
        # ])
        
        # fig.update_layout(
        #     hovermode="x",
        #     showlegend=False,
        #     title_x=0.5,
        #     font=dict(
        #     family="Courier New, monospace",
        #     size=20
        #     )
        # )
        # fig.add_annotation(x=200, y=2.5, text="μ: {}<br>σ: {}".format(df["mean"].mean().round(2), df["mean"].std().round(2)), showarrow=False)
        # fig.update_yaxes(range=[-1.5, 4])

        # fig.show()
    elif args.command == "intention":
        data_files = next(os.walk(args.path))[2]
        for i in range(0, len(data_files), 3):
            data_files = sorted(data_files)
            behaviour_data_files = data_files[i:i+3]
            dfs = []
            read_first = False
            cols = ["Step", "Value"]
            for behaviour_data_f in behaviour_data_files:
                if read_first:
                    cols = ["Value"]
                df = pd.read_csv(os.path.join(args.path, behaviour_data_f), usecols=cols)
                df.rename(columns={"Value": ''.join(behaviour_data_f.split('.')[0].split('_')[-2:])}, inplace=True)
                dfs.append(df)
                read_first = True
            df = pd.concat(dfs, axis=1)
            fig = px.line(df, x="Step", y=df.columns)
            fig.show()
    elif args.command == "success":
        data_files = next(os.walk(args.path))[2]
        if args.noise:
            dirs = ["noise_5", "noise_10"]
            data_files = []
            for dir_name in dirs:
                files = next(os.walk(os.path.join(args.path, dir_name)))[2]
                files = list(map(lambda s: dir_name + '/' + s, files))
                data_files.extend(files)
            data_files = sorted(data_files)
        
        dfs = []
        read_first = False
        cols = ["Step", "Value"]
        for i in range(len(data_files)):
            if data_files[i].endswith(".csv"):
                if read_first:
                    cols = ["Value"]
                data_f = data_files[i]
                df = pd.read_csv(os.path.join(args.path, data_f), usecols=cols)
                if not args.noise:
                    df.rename(columns={"Value": "success_{}".format(i)}, inplace=True)
                else:
                    col_name = data_f.split('/')[0] + "_success_{}".format(i%5)
                    df.rename(columns={"Value": col_name}, inplace=True)
                dfs.append(df)
                read_first = True
        df = pd.concat(dfs, axis=1)
        
        if not args.noise:
            df["mean"] = df.loc[:, "success_0":"success_9"].mean(axis=1)
            df["std"] = df.loc[:, "success_0":"success_9"].std(axis=1).round(2)
        else:
            for dir_name in dirs:
                df[dir_name + "_mean"] = df.loc[:, dir_name + "_success_0":dir_name + "_success_4"].mean(axis=1)
                df[dir_name + "_std"] = df.loc[:, dir_name + "_success_0":dir_name + "_success_4"].std(axis=1).round(2)
        # if not args.noise:
        #     fig = go.Figure([
        #         go.Scatter(
        #             x=df["Step"],
        #             y=df["mean"],
        #             mode="lines",
        #             marker=dict(color="blue"),
        #             showlegend=False
        #         ),
        #         go.Scatter(
        #             x=df["Step"],
        #             y=df["mean"]+df["std"],
        #             mode="lines",
        #             marker=dict(color="#444"),
        #             line=dict(width=0),
        #             showlegend=False
        #         ),
        #         go.Scatter(
        #             x=df["Step"],
        #             y=df["mean"]-df["std"],
        #             marker=dict(color="#444"),
        #             line=dict(width=0),
        #             mode="lines",
        #             fillcolor='rgba(68, 68, 68, 0.3)',
        #             fill='tonexty',
        #             showlegend=False
        #         )
        #     ])
        # else:
        #     fig = go.Figure([
        #         go.Scatter(
        #             name="5% Noise",
        #             x=df["Step"],
        #             y=df["noise_5_mean"],
        #             mode="lines",
        #             marker=dict(color="blue"),
        #             showlegend=False
        #         ),
        #         go.Scatter(
        #             x=df["Step"],
        #             y=df["noise_5_mean"]+df["noise_5_std"],
        #             mode="lines",
        #             marker=dict(color="#444"),
        #             line=dict(width=0),
        #             showlegend=False
        #         ),
        #         go.Scatter(
        #             x=df["Step"],
        #             y=df["noise_5_mean"]-df["noise_5_std"],
        #             marker=dict(color="#444"),
        #             line=dict(width=0),
        #             mode="lines",
        #             fillcolor='rgba(68, 68, 68, 0.3)',
        #             fill='tonexty',
        #             showlegend=False
        #         ),
        #         go.Scatter(
        #             name="10% Noise",
        #             x=df["Step"],
        #             y=df["noise_10_mean"],
        #             mode="lines",
        #             marker=dict(color="purple"),
        #             showlegend=False
        #         ),
        #         go.Scatter(
        #             x=df["Step"],
        #             y=df["noise_10_mean"]+df["noise_10_std"],
        #             mode="lines",
        #             marker=dict(color="#444"),
        #             line=dict(width=0),
        #             showlegend=False
        #         ),
        #         go.Scatter(
        #             x=df["Step"],
        #             y=df["noise_10_mean"]-df["noise_10_std"],
        #             marker=dict(color="#444"),
        #             line=dict(width=0),
        #             mode="lines",
        #             fillcolor='rgba(0, 68, 0, 0.3)',
        #             fill='tonexty',
        #             showlegend=False
        #         ),
        #     ])
        # fig.update_layout(
        #     title_text="Sampling From Latent Space",
        #     hovermode="x",
        #     xaxis_title="Episodes",
        #     yaxis_title="Success",
        #     title_x=0.5,
        #     showlegend=True,
        #     font=dict(
        #         family="Courier New, monospace",
        #         size=20
        #     )
        # )
        # fig.add_annotation(x=df["Step"].iloc[-1], y=df["mean"].iloc[-1], text=df["mean"].iloc[-1], arrowhead=2, textangle=0, ayref='y', ax=1, ay=90)
        # # fig.add_annotation(x=df["Step"].iloc[-1], y=df["noise_10_mean"].iloc[-1], text=df["noise_10_mean"].iloc[-1], arrowhead=2, textangle=0, ayref='y', ax=1, ay=35)
        # fig.update_yaxes(range=[0,100])
        # fig.show()