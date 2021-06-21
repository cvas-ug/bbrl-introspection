import gym
import numpy as np
import torch
import torch.optim as optim
import torch.cuda
import time
import csv
import os
import json
import torch.nn as nn

from itertools import count
from models import BehaviourNetwork
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def get_behaviour_from_model_output(behaviour_means, behaviour_log_stds):
    num = len(behaviour_means)
    act = torch.zeros(num).type(torch.cuda.FloatTensor)
    for i in range(num):
        normal = Normal(behaviour_means[i], behaviour_log_stds[i])
        X = normal.rsample()
        act[i] = torch.tanh(X)
    return act

def train(rank, args, shared_model, counter, lock, optimizer=None):
    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    
    env = gym.make("FetchPickAndPlace-v1")
    flattened_env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

    model = BehaviourNetwork(args.weights_path, args.command)
    writer = SummaryWriter("experiments/retract")
    if args.use_cuda:
        model.cuda()
    torch.cuda.manual_seed_all(12)
    
    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    done = True
    losses = []
    for num_iter in count():
        with lock:
            counter.value += 1
        obs = env.reset()
        timestep = 0 #count the total number of timesteps
        if rank == 0:

            if num_iter % args.save_interval == 0 and num_iter > 0:
                shared_model.save_model_weights()
        if num_iter % (args.save_interval * 2.5) == 0 and num_iter > 0 and rank == 1:    # Second saver in-case first processes crashes 
            shared_model.save_model_weights()
        model.load_state_dict(shared_model.state_dict())
        criterion = nn.MSELoss()

        object_oriented_goal = obs['observation'][6:9]
        object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object
        while np.linalg.norm(object_oriented_goal) >= 0.015 and timestep <= env._max_episode_steps:
            action = [0, 0, 0, 0, 0]
            
            for i in range(len(object_oriented_goal)):
                action[i] = object_oriented_goal[i]*6

            action[3] = 0.05 #open

            obs, reward, done, info = env.step(action)
            timestep += 1

            object_oriented_goal = obs['observation'][6:9]
            object_oriented_goal[2] += 0.03
        object_oriented_goal = obs['observation'][6:9]
        while np.linalg.norm(object_oriented_goal) >= 0.005 and timestep <= env._max_episode_steps :
            action = [0, 0, 0, 0, 0]
            for i in range(len(object_oriented_goal)):
                action[i] = object_oriented_goal[i]*6

            action[3] = -0.01
            action[4] = obs['observation'][13]/8
            obs, reward, done, info = env.step(action)
            timestep += 1

            object_oriented_goal = obs['observation'][6:9]

        state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
        goal = obs['desired_goal']
        objectPos = obs['observation'][3:6]
        losses = []
        while np.linalg.norm(goal - objectPos) >= 0.01 and timestep <= env._max_episode_steps :

            action = [0, 0, 0, 0, 0]
            _, output = model(state_inp)
            act_tensor = get_behaviour_from_model_output(output["retract_means"], output["retract_log_stds"])
            loss = 0
            for i in range(len(goal - objectPos)):
                optimizer.zero_grad()
                expected = torch.from_numpy(np.array((goal - objectPos)[i]*6)).type(FloatTensor)
                action[i] = act_tensor[i].cpu().detach().numpy()
                error = criterion(act_tensor[i], expected)
                (error).backward(retain_graph=True)
                loss += error.cpu().detach().item()
                ensure_shared_grads(model, shared_model)
                optimizer.step()
            losses.append(loss)
            action[3]= -0.01
            obs, reward, done, info = env.step(action)
            timestep += 1
            objectPos = obs['observation'][3:6]
            state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor) 
        writer.add_scalar("loss", np.mean(losses), num_iter)
        while True: #limit the number of timesteps in the episode to a fixed duration
            
            action = [0, 0, 0, 0, 0]
            action[3] = -0.01 # keep the gripper closed

            obs, reward, done, info = env.step(action)
            timestep += 1

            if timestep >= env._max_episode_steps: break

def test(rank, args, shared_model, counter):
    
    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    env = gym.make("FetchPickAndPlace-v1")
    flattened_env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

    model = BehaviourNetwork(args.weights_path, args.command)
    if args.use_cuda:
        model.cuda()
    model.eval()
    done = True       
    writer = SummaryWriter("experiments/retract")

    total_eps = 0
    while True:
        model.load_state_dict(shared_model.state_dict())
        ep_num = 0
        num_ep = counter.value
        success = 0
        while ep_num < 100:
            ep_num +=1            
            obs = env.reset()
            
            timestep = 0
            model.load_state_dict(shared_model.state_dict())
            
            object_oriented_goal = obs['observation'][6:9]
            object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object
            while np.linalg.norm(object_oriented_goal) >= 0.015 and timestep <= env._max_episode_steps:
                # env.render()
                action = [0, 0, 0, 0, 0]

                for i in range(len(object_oriented_goal)):
                    action[i] = object_oriented_goal[i]*6

                action[3] = 0.05 #open

                obs, reward, done, info = env.step(action)
                timestep += 1

                object_oriented_goal = obs['observation'][6:9]
            object_oriented_goal = obs['observation'][6:9]
            while np.linalg.norm(object_oriented_goal) >= 0.005 and timestep <= env._max_episode_steps :
                # env.render()
                action = [0, 0, 0, 0, 0]
                for i in range(len(object_oriented_goal)):
                    action[i] = object_oriented_goal[i]*6
                
                action[3] = -0.01
                action[4] = obs['observation'][13]/8
                obs, reward, done, info = env.step(action)
                timestep += 1

                object_oriented_goal = obs['observation'][6:9]
            
            goal = obs['desired_goal']
            objectPos = obs['observation'][3:6]
            state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
            while np.linalg.norm(goal - objectPos) >= 0.01 and timestep <= env._max_episode_steps :
                # env.render()
                action = [0, 0, 0, 0, 0]
                _, output = model(state_inp)
                act_tensor = get_behaviour_from_model_output(output["retract_means"], output["retract_log_stds"])
                for i in range(3):
                    action[i] = act_tensor[i].cpu().detach().numpy()
                
                action[3]= -0.01
                obs, reward, done, info = env.step(action)
                timestep += 1
                state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
                objectPos = obs['observation'][3:6]
                if timestep >= env._max_episode_steps: break

            while True: #limit the number of timesteps in the episode to a fixed duration
                # env.render()
                action = [0, 0, 0, 0, 0]
                action[3] = -0.01 # keep the gripper closed

                obs, reward, done, info = env.step(action)
                timestep += 1

                if timestep >= env._max_episode_steps: break
            
            if info['is_success'] == 1.0:
                success +=1
            if done:
                if ep_num % 100==0:
                    total_eps += 100        
                    print("num episodes {}, success {}".format(total_eps, success))
                    writer.add_scalar("success", success, total_eps)