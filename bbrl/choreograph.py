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
import torch.nn.functional as F

from itertools import count
from models import BehaviourNetwork, ChoreographNetwork
from torch.distributions import Normal
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def get_behaviour_from_model_output(model_output, behaviour):
    # 0 -> approach, 1 -> manipulate, 2 -> retract
    if behaviour == 0:
        behaviour_means = model_output["approach_means"]
        behaviour_log_stds = model_output["approach_log_stds"]
    elif behaviour == 1:
        behaviour_means = model_output["grasp_means"]
        behaviour_log_stds = model_output["grasp_log_stds"]
    else:
        behaviour_means = model_output["retract_means"]
        behaviour_log_stds = model_output["retract_log_stds"]
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

    behaviour_net = BehaviourNetwork(args.weights_path)
    model = ChoreographNetwork(args.weights_path, internal_states=True)
    # model = ChoreographNetwork(args.weights_path)
    writer = SummaryWriter("experiments/choreograph/means_logvar_unfrozen/run_5")
    torch.cuda.manual_seed_all(29)
    if args.use_cuda:
        model.cuda()
        behaviour_net.cuda()
        
    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
    
    model.train()
    done = True
    a = 0
    success = 0
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
        state_values, log_probs, rewards, entropies = [], [], [], []
        if done:
            cx = Variable(torch.zeros(1, 32)).type(FloatTensor)
            hx = Variable(torch.zeros(1, 32)).type(FloatTensor)
        else:
            cx = Variable(cx.data).type(FloatTensor)
            hx = Variable(hx.data).type(FloatTensor)

        state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
        # noise = Normal(0, 0.1).samplmodel
        output = model(state_inp, hx, cx)
        state_value = output["state"]
        action_values = output["actions"]
        hx = output["hidden"]
        cx = output["cell"]
        prob = F.softmax(action_values, dim=-1)
        log_prob = F.log_softmax(action_values, dim=-1)
        behaviour = prob.max(-1, keepdim=True)[1].data
        entropy = -(log_prob * prob).sum(-1, keepdim=True)
        log_prob = log_prob.gather(-1, Variable(behaviour))
        action_out = behaviour.to(torch.device("cpu"))
        writer.add_scalar("approach_state_value", state_value.cpu().detach().item(), num_iter)
        entropies.append(entropy), log_probs.append(log_prob), state_values.append(state_value)
        
        object_oriented_goal = obs['observation'][6:9]
        object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object
        while np.linalg.norm(object_oriented_goal) >= 0.015 and timestep <= env._max_episode_steps:
            
            action = [0, 0, 0, 0, 0]
            _, output = behaviour_net(state_inp)
            act_tensor = get_behaviour_from_model_output(output, action_out)
            for i in range(len(object_oriented_goal)):
                action[i] = act_tensor[i].cpu().detach().numpy()
            
            action[3] = 0.05
            obs, reward, done, info = env.step(action)
            timestep += 1
            object_oriented_goal = obs['observation'][6:9]
            object_oriented_goal[2] += 0.03
            state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
            # noise = Normal(0, 0.1).sample(sample_shape=state_inp.size()).type(FloatTensor)
            # state_inp = state_inp + (args.noise / 100) * noise
            if timestep >= env._max_episode_steps: 
                reward = torch.Tensor([-1.0]).type(FloatTensor)
                break
        
        if timestep < env._max_episode_steps: 
            reward = torch.Tensor([1.0]).type(FloatTensor)
        rewards.append(reward)

        output = model(state_inp, hx, cx)
        state_value = output["state"]
        action_values = output["actions"]
        hx = output["hidden"]
        cx = output["cell"]
        
        prob = F.softmax(action_values, dim=-1)
        log_prob = F.log_softmax(action_values, dim=-1)
        behaviour = prob.max(-1, keepdim=True)[1].data
        entropy = -(log_prob * prob).sum(-1, keepdim=True)
        log_prob = log_prob.gather(-1, Variable(behaviour))
        action_out = behaviour.to(torch.device("cpu"))
        writer.add_scalar("grasp_state_value", state_value.cpu().detach().item(), num_iter)
        entropies.append(entropy), log_probs.append(log_prob), state_values.append(state_value)
        object_oriented_goal = obs['observation'][6:9]
        while np.linalg.norm(object_oriented_goal) >= 0.005 and timestep <= env._max_episode_steps :
            action = [0, 0, 0, 0, 0]
            _, output = behaviour_net(state_inp)
            act_tensor = get_behaviour_from_model_output(output, action_out)
            for i in range(len(object_oriented_goal)):
                action[i] = act_tensor[i].cpu().detach().numpy()
            
            action[3]= -0.01 
            if action_out == 1:
                action[4] = act_tensor[3].cpu().detach().numpy()
            
            obs, reward, done, info = env.step(action)
            timestep += 1

            object_oriented_goal = obs['observation'][6:9]
            state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
            # noise = Normal(0, 0.1).sample(sample_shape=state_inp.size()).type(FloatTensor)
            # state_inp = state_inp + (args.noise / 100) * noise
            if timestep >= env._max_episode_steps: 
                reward = torch.Tensor([-1.0]).type(FloatTensor)
                break
        
        if timestep < env._max_episode_steps:
            reward = torch.Tensor([1.0]).type(FloatTensor)
        rewards.append(reward)

        output = model(state_inp, hx, cx)
        state_value = output["state"]
        action_values = output["actions"]
        hx = output["hidden"]
        cx = output["cell"]
        
        prob = F.softmax(action_values, dim=-1)
        log_prob = F.log_softmax(action_values, dim=-1)
        behaviour = prob.max(-1, keepdim=True)[1].data
        entropy = -(log_prob * prob).sum(-1, keepdim=True)
        log_prob = log_prob.gather(-1, Variable(behaviour))
        action_out = behaviour.to(torch.device("cpu"))
        writer.add_scalar("retract_state_value", state_value.cpu().detach().item(), num_iter)
        entropies.append(entropy), log_probs.append(log_prob), state_values.append(state_value)
        goal = obs['desired_goal']
        objectPos = obs['observation'][3:6]
        while np.linalg.norm(goal - objectPos) >= 0.01 and timestep <= env._max_episode_steps :
            
            action = [0, 0, 0, 0, 0]
            _, output = behaviour_net(state_inp)
            act_tensor = get_behaviour_from_model_output(output, action_out)

            for i in range(len(goal - objectPos)):
                action[i] = act_tensor[i].cpu().detach().numpy()
            
            action[3] = -0.01
            obs, reward, done, info = env.step(action)
            timestep += 1
            state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
            # noise = Normal(0, 0.1).sample(sample_shape=state_inp.size()).type(FloatTensor)
            # state_inp = state_inp + (args.noise / 100) * noise
            objectPos = obs['observation'][3:6]

            if timestep >= env._max_episode_steps: 
                break

        while True: #limit the number of timesteps in the episode to a fixed duration
            action = [0, 0, 0, 0, 0]
            action[3] = -0.01 # keep the gripper closed

            obs, reward, done, info = env.step(action)
            timestep += 1

            if timestep >= env._max_episode_steps: break
        
        if info['is_success'] == 1.0:
            reward = torch.Tensor([1.0]).type(FloatTensor)
            success += 1
        else:
            reward = torch.Tensor([-1.0]).type(FloatTensor)
        rewards.append(reward)
        
        R = torch.zeros(1, 1)
        state_values.append(Variable(R).type(FloatTensor))
        policy_loss = 0
        value_loss = 0
        R = Variable(R).type(FloatTensor)
        gae = torch.zeros(1, 1).type(FloatTensor)
        
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - state_values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            delta_t = rewards[i] + args.gamma * \
                state_values[i + 1].data - state_values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae).type(FloatTensor)

        total_loss = policy_loss + args.value_loss_coef * value_loss
        optimizer.zero_grad()
        (total_loss).backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        writer.add_scalar("policy_loss", policy_loss.cpu().detach().item(), num_iter)
        writer.add_scalar("value_loss", value_loss.cpu().detach().item(), num_iter)
        writer.add_scalar("total_loss", total_loss.cpu().detach().item(), num_iter)
        ensure_shared_grads(model, shared_model)
        optimizer.step()

def test(rank, args, shared_model, counter):
    
    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    env = gym.make("FetchPickAndPlace-v1")
    flattened_env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

    behaviour_net = BehaviourNetwork(args.weights_path)
    model = ChoreographNetwork(args.weights_path, internal_states=True)
    # model = ChoreographNetwork(args.weights_path)
    
    writer = SummaryWriter("experiments/choreograph/means_logvar_unfrozen/run_5")
    if args.use_cuda:
        model.cuda()
        behaviour_net.cuda()

    done = True

    total_eps = 0
    while True:
        model.load_state_dict(shared_model.state_dict())
        model.eval()
        behaviour_net.eval()
        ep_num = 1
        success = 0
        num_ep = counter.value
        while ep_num < 100:
            ep_num +=1            
            obs = env.reset()
            timestep = 0
            if done:
                cx = Variable(torch.zeros(1, 32)).type(FloatTensor)
                hx = Variable(torch.zeros(1, 32)).type(FloatTensor)
            else:
                cx = Variable(cx.data).type(FloatTensor)
                hx = Variable(hx.data).type(FloatTensor)
            state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
            # noise = Normal(0, 0.1).sample(sample_shape=state_inp.size()).type(FloatTensor)
            # state_inp = state_inp + (args.noise / 100) * noise
            output = model(state_inp, hx, cx)
            state_value = output["state"]
            action_values = output["actions"]
            hx = output["hidden"]
            cx = output["cell"]

            prob = F.softmax(action_values, dim=-1)
            behaviour = prob.max(-1, keepdim=True)[1].data
            action_out = behaviour.to(torch.device("cpu"))
            object_oriented_goal = obs['observation'][6:9]
            object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object
            while np.linalg.norm(object_oriented_goal) >= 0.015 and timestep <= env._max_episode_steps:
                # env.render()
                action = [0, 0, 0, 0, 0]
                _, output = behaviour_net(state_inp)
                act_tensor = get_behaviour_from_model_output(output, action_out)
                
                for i in range(len(object_oriented_goal)):
                    action[i] = act_tensor[i].cpu().detach().numpy()
                
                action[3] = 0.05
                obs, reward, done, info = env.step(action)
                timestep += 1
                object_oriented_goal = obs['observation'][6:9]
                object_oriented_goal[2] += 0.03
                state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
                # noise = Normal(0, 0.1).sample(sample_shape=state_inp.size()).type(FloatTensor)
                # state_inp = state_inp + (args.noise / 100) * noise
                if timestep >= env._max_episode_steps: 
                    break

            output = model(state_inp, hx, cx)
            state_value = output["state"]
            action_values = output["actions"]
            hx = output["hidden"]
            cx = output["cell"]

            prob = F.softmax(action_values, dim=-1)
            behaviour = prob.max(-1, keepdim=True)[1].data
            action_out = behaviour.to(torch.device("cpu"))
            object_oriented_goal = obs['observation'][6:9]
            while np.linalg.norm(object_oriented_goal) >= 0.005 and timestep <= env._max_episode_steps:
                # env.render()
                action = [0, 0, 0, 0, 0]
                _, output = behaviour_net(state_inp)
                act_tensor = get_behaviour_from_model_output(output, action_out)
                
                for i in range(len(object_oriented_goal)):
                    action[i] = act_tensor[i].cpu().detach().numpy()
                
                action[3]= -0.01 
                if action_out == 1:
                    action[4] = act_tensor[3].cpu().detach().numpy()
                
                obs, reward, done, info = env.step(action)
                timestep += 1

                object_oriented_goal = obs['observation'][6:9]
                state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
                # noise = Normal(0, 0.1).sample(sample_shape=state_inp.size()).type(FloatTensor)
                # state_inp = state_inp + (args.noise / 100) * noise
                if timestep >= env._max_episode_steps: 
                    break

            output = model(state_inp, hx, cx)
            state_value = output["state"]
            action_values = output["actions"]
            hx = output["hidden"]
            cx = output["cell"]

            prob = F.softmax(action_values, dim=-1)
            behaviour = prob.max(-1, keepdim=True)[1].data
            action_out = behaviour.to(torch.device("cpu"))
            goal = obs['desired_goal']
            objectPos = obs['observation'][3:6]
            while np.linalg.norm(goal - objectPos) >= 0.01 and timestep <= env._max_episode_steps:
                # env.render()
            
                action = [0, 0, 0, 0, 0]
                _, output = behaviour_net(state_inp)
                act_tensor = get_behaviour_from_model_output(output, action_out)

                for i in range(len(goal - objectPos)):
                    action[i] = act_tensor[i].cpu().detach().numpy()
                
                action[3] = -0.01
                obs, reward, done, info = env.step(action)
                timestep += 1
                state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
                # noise = Normal(0, 0.1).sample(sample_shape=state_inp.size()).type(FloatTensor)
                # state_inp = state_inp + (args.noise / 100) * noise
                objectPos = obs['observation'][3:6]
                object_rel_pos = obs['observation'][6:9]
                if timestep >= env._max_episode_steps: 
                    break

            while True: #limit the number of timesteps in the episode to a fixed duration
                # env.render()
                action = [0, 0, 0, 0, 0]
                action[3] = -0.01 # keep the gripper closed

                obs, reward, done, info = env.step(action)
                timestep += 1

                if timestep >= env._max_episode_steps: break
            if info['is_success'] == 1.0:
                success +=1
            if ep_num % 100==0:
                total_eps += 100
                print("num episodes {}, success {}".format(total_eps, success))
                writer.add_scalar("test_succes", success, total_eps)