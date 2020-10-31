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

def get_behaviour_from_model_output(behaviour_means, behaviour_log_stds):
    num = len(behaviour_means)
    act = torch.zeros(num).type(torch.cuda.FloatTensor)
    for i in range(num):
        normal = Normal(behaviour_means[i], behaviour_log_stds[i].exp())
        X = normal.rsample()
        act[i] = torch.tanh(X)
    return act

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, args, shared_model, counter, lock, optimizer=None):
    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    
    env = gym.make("FetchPickAndPlace-v1")
    flattened_env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

    model = BehaviourNetwork("weights", args.command)
    
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
        
        timeStep = 0 #count the total number of timesteps
        if rank == 0:
            if num_iter % args.save_interval == 0 and num_iter > 0:
                shared_model.save_model_weights()
        if num_iter % (args.save_interval * 2.5) == 0 and num_iter > 0 and rank == 1:    # Second saver in-case first processes crashes 
            shared_model.save_model_weights()
        model.load_state_dict(shared_model.state_dict())
        state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
        criterion = nn.MSELoss()
        object_oriented_goal = obs['observation'][6:9]
        object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object
        while np.linalg.norm(object_oriented_goal) >= 0.015 and timeStep <= env._max_episode_steps:
            
            action = [0, 0, 0, 0, 0]
            error = torch.zeros(3).type(FloatTensor) 
            _, output = model(state_inp)
            act_tensor = get_behaviour_from_model_output(output["approach_means"], output["approach_log_stds"])
            optimizer.zero_grad()                   
            for i in range(len(object_oriented_goal)):
                action[i] = act_tensor[i].cpu().detach().numpy()
                expected = torch.from_numpy(np.array(object_oriented_goal[i]*6)).type(FloatTensor)                  
                error[i] = criterion(act_tensor[i], expected)

            loss = torch.sum(error)
            loss.backward(retain_graph=True)
            losses.append(loss.cpu().detach().item())
            
            ensure_shared_grads(model, shared_model)
            optimizer.step()
           
            action[3] = 0.05
            obs, reward, done, info = env.step(action)
            timeStep += 1
            object_oriented_goal = obs['observation'][6:9]
            object_oriented_goal[2] += 0.03
            state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
            if timeStep >= env._max_episode_steps: break
        object_oriented_goal = obs['observation'][6:9]
        while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env._max_episode_steps :
           
            action = [0, 0, 0, 0, 0]
            for i in range(len(object_oriented_goal)):
                action[i] = object_oriented_goal[i]*6

            action[4] = obs['observation'][13]/6
            action[3] = -0.02

            obs, reward, done, info = env.step(action)
            timeStep += 1

            object_oriented_goal = obs['observation'][6:9]
            if timeStep >= env._max_episode_steps: break
        goal = obs['desired_goal']
        objectPos = obs['observation'][3:6]
        while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps :
            
            action = [0, 0, 0, 0, 0]
            for i in range(len(goal - objectPos)):
                action[i] = (goal - objectPos)[i]*6

            action[3] = -0.01
            obs, reward, done, info = env.step(action)
            timeStep += 1

            objectPos = obs['observation'][3:6]
            if timeStep >= env._max_episode_steps: break
        
        while True: #limit the number of timesteps in the episode to a fixed duration
            
            action = [0, 0, 0, 0, 0]
            action[3] = -0.01 # keep the gripper closed

            obs, reward, done, info = env.step(action)
            timeStep += 1

            if timeStep >= env._max_episode_steps: break

def test(rank, args, shared_model, counter):
    
    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    env = gym.make("FetchPickAndPlace-v1")
    flattened_env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

    model = BehaviourNetwork("weights", args.command)
    if args.use_cuda:
        model.cuda()
    model.eval()
    done = True
    writer = SummaryWriter("experiments/approach")
    
    total_eps = 0
    while True:
        model.load_state_dict(shared_model.state_dict())
        ep_num = 1
        num_ep = counter.value
        success = 0
        while ep_num <= 100:
            ep_num +=1            
            obs = env.reset()
        
            timeStep = 0
            state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
            model.load_state_dict(shared_model.state_dict())
            object_oriented_goal = obs['observation'][6:9]
            object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object
            while np.linalg.norm(object_oriented_goal) >= 0.015 and timeStep <= env._max_episode_steps:
                action = [0, 0, 0, 0, 0]

                _, output = model(state_inp)
                act_tensor = get_behaviour_from_model_output(output["approach_means"], output["approach_log_stds"])
                
                for i in range(len(object_oriented_goal)):
                    action[i] = act_tensor[i].cpu().detach().numpy()
                
                action[3] = 0.05
                obs, reward, done, info = env.step(action)
                timeStep += 1
                object_oriented_goal = obs['observation'][6:9]
                object_oriented_goal[2] += 0.03
                state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
                if timeStep >= env._max_episode_steps: break
            object_oriented_goal = obs['observation'][6:9]
            while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env._max_episode_steps :
                #env.render()
                action = [0, 0, 0, 0, 0]
                for i in range(len(object_oriented_goal)):
                    action[i] = object_oriented_goal[i]*6

                action[4] = obs['observation'][13]/8
                action[3] = -0.02

                obs, reward, done, info = env.step(action)
                timeStep += 1

                object_oriented_goal = obs['observation'][6:9]
                if timeStep >= env._max_episode_steps: break

            goal = obs['desired_goal']
            objectPos = obs['observation'][3:6]
            while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps :
                #env.render()
                action = [0, 0, 0, 0, 0]
                for i in range(len(goal - objectPos)):
                    action[i] = (goal - objectPos)[i]*6

                action[3] = -0.01
                obs, reward, done, info = env.step(action)
                timeStep += 1

                objectPos = obs['observation'][3:6]
                if timeStep >= env._max_episode_steps: break
                    
            while True: #limit the number of timesteps in the episode to a fixed duration
                #env.render()
                action = [0, 0, 0, 0, 0]
                action[3] = -0.01 # keep the gripper closed

                obs, reward, done, info = env.step(action)
                timeStep += 1

                if timeStep >= env._max_episode_steps: break
            
            if info['is_success'] == 1.0:
                success +=1
            if done:
                if ep_num % 100==0:
                    total_eps += 100        
                    print("num episodes {}, success {}".format(total_eps, success))
                    writer.add_scalar("success", success, total_eps)