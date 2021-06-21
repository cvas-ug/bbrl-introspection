import gym
import random
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from torch.autograd import Variable
from models import BehaviourNetwork, ChoreographNetwork

activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def get_all_model_outputs(model_out):
    output = {"approach": [], "grasp": [], "retract": []}
    behaviour_keys = {"approach": ["approach_means", "approach_log_stds"], "grasp": ["grasp_means", "grasp_log_stds"], "retract": ["retract_means", "retract_log_stds"]}
    
    for key in sorted(behaviour_keys):
        behaviour = behaviour_keys[key]
        behaviour_means = model_out[behaviour[0]]
        behaviour_log_stds = model_out[behaviour[1]]
        for i in range(len(behaviour_means)):
            normal = Normal(behaviour_means[i], behaviour_log_stds[i])
            X = normal.rsample()
            output[key].append(torch.tanh(X))

    return output

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

env = gym.make("FetchPickAndPlace-v1")
env.seed(seed=1)
flattened_env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--use-cuda',default=True,
                    help='run on gpu.')
parser.add_argument("--noise", type=int)
args = parser.parse_args() 
writer = SummaryWriter("experiments/test")
behaviour_net = BehaviourNetwork("./")
choreographer = ChoreographNetwork("./")
if args.use_cuda:
    behaviour_net.cuda()
    choreographer.cuda()
torch.cuda.manual_seed_all(12)

FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor

behaviour_net.eval()
choreographer.eval()

max_eps = 2000
max_steps = 50
ep_num = 0
done = True
success = 0         

behaviour_net.feature_net.fc1.register_forward_hook(get_activation("fc1"))
behaviour_net.feature_net.fc2.register_forward_hook(get_activation("fc2"))
acts_dict = {"approach": {}, "grasp": {}, "retract": {}}
while ep_num < max_eps:
    
    for key in acts_dict.keys():
        acts_dict[key][ep_num] = []
    obs = env.reset()
    
    timestep = 0 #count the total number of timesteps
    state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
    # noise = Normal(0, 0.1).sample(sample_shape=state_inp.size()).type(FloatTensor)
    # state_inp = state_inp + (args.noise / 100) * noise
    if done:
        cx = Variable(torch.zeros(1, 32)).type(FloatTensor)
        hx = Variable(torch.zeros(1, 32)).type(FloatTensor)
    else:
        cx = Variable(cx.data).type(FloatTensor)
        hx = Variable(hx.data).type(FloatTensor)

    output = choreographer(state_inp, hx, cx)
    state_value = output["state"]
    action_values = output["actions"]
    hx = output["hidden"]
    cx = output["cell"]

    prob = F.softmax(action_values, dim=-1)
    
    act_model = prob.max(-1, keepdim=True)[1].data
    
    action_out = act_model.to(torch.device("cpu"))
    object_oriented_goal = obs['observation'][6:9]
    object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object
    while np.linalg.norm(object_oriented_goal) >= 0.015 and timestep <= env._max_episode_steps:
        # env.render()
        action = [0, 0, 0, 0, 0]
        _, output = behaviour_net(state_inp)
        if ep_num == max_eps - 1:
            writer.add_scalar("grip_x", obs["observation"][0], timestep)
            writer.add_scalar("grip_y", obs["observation"][1], timestep)
            writer.add_scalar("grip_z", obs["observation"][2], timestep)
            writer.add_scalar("obj_x", obs["observation"][3], timestep)
            writer.add_scalar("obj_y", obs["observation"][4], timestep)
            writer.add_scalar("obj_z", obs["observation"][5], timestep)
            writer.add_scalar("obj_rel_grip_x", obs["observation"][6], timestep)
            writer.add_scalar("obj_rel_grip_y", obs["observation"][7], timestep)
            writer.add_scalar("obj_rel_grip_z", obs["observation"][8], timestep)
            writer.add_scalar("grip_state_1", obs["observation"][9], timestep)
            writer.add_scalar("grip_state_2", obs["observation"][10], timestep)
            writer.add_scalar("obj_rotx", obs["observation"][11], timestep)
            writer.add_scalar("obj_roty", obs["observation"][12], timestep)
            writer.add_scalar("obj_rotz", obs["observation"][13], timestep)
            writer.add_scalar("obj_velx", obs["observation"][14], timestep)
            writer.add_scalar("obj_vely", obs["observation"][15], timestep)
            writer.add_scalar("obj_velz", obs["observation"][16], timestep)
            writer.add_scalar("obj_velrx", obs["observation"][17], timestep)
            writer.add_scalar("obj_velry", obs["observation"][18], timestep)
            writer.add_scalar("obj_velrz", obs["observation"][19], timestep)
            writer.add_scalar("grip_velx", obs["observation"][20], timestep)
            writer.add_scalar("grip_vely", obs["observation"][21], timestep)
            writer.add_scalar("grip_velz", obs["observation"][22], timestep)
            writer.add_scalar("grip_vel_1", obs["observation"][23], timestep)
            writer.add_scalar("grip_vel_2", obs["observation"][24], timestep)
        timestep_dict = {}
        for key in ["fc1", "fc2"]:
            acts_tensor = activations[key]
            acts_tensor = F.elu(acts_tensor)
            timestep_dict[key] = acts_tensor.cpu()
        acts_dict["approach"][ep_num].append(timestep_dict)
        act_tensor = get_behaviour_from_model_output(output, action_out)
        
        for i in range(3):
            action[i] = act_tensor[i].cpu().detach().numpy()
        
        action[3] = 0.05
        obs, reward, done, info = env.step(action)
        timestep += 1
        object_oriented_goal = obs['observation'][6:9]
        object_oriented_goal[2] += 0.03
        state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
        # noise = Normal(0, 0.1).sample(sample_shape=state_inp.size()).type(FloatTensor)
        # state_inp = state_inp + (args.noise / 100) * noise
        
        if timestep >= env._max_episode_steps: break
    output = choreographer(state_inp, hx, cx)
    state_value = output["state"]
    action_values = output["actions"]
    hx = output["hidden"]
    cx = output["cell"]
    prob = F.softmax(action_values, dim=-1)
    act_model = prob.max(-1, keepdim=True)[1].data
    action_out = act_model.to(torch.device("cpu"))
    object_oriented_goal = obs['observation'][6:9]
    while np.linalg.norm(object_oriented_goal) >= 0.005 and timestep <= env._max_episode_steps:
        # env.render()
        action = [0, 0, 0, 0, 0]
        _, output = behaviour_net(state_inp)
        if ep_num == max_eps - 1:
            writer.add_scalar("grip_x", obs["observation"][0], timestep)
            writer.add_scalar("grip_y", obs["observation"][1], timestep)
            writer.add_scalar("grip_z", obs["observation"][2], timestep)
            writer.add_scalar("obj_x", obs["observation"][3], timestep)
            writer.add_scalar("obj_y", obs["observation"][4], timestep)
            writer.add_scalar("obj_z", obs["observation"][5], timestep)
            writer.add_scalar("obj_rel_grip_x", obs["observation"][6], timestep)
            writer.add_scalar("obj_rel_grip_y", obs["observation"][7], timestep)
            writer.add_scalar("obj_rel_grip_z", obs["observation"][8], timestep)
            writer.add_scalar("grip_state_1", obs["observation"][9], timestep)
            writer.add_scalar("grip_state_2", obs["observation"][10], timestep)
            writer.add_scalar("obj_rotx", obs["observation"][11], timestep)
            writer.add_scalar("obj_roty", obs["observation"][12], timestep)
            writer.add_scalar("obj_rotz", obs["observation"][13], timestep)
            writer.add_scalar("obj_velx", obs["observation"][14], timestep)
            writer.add_scalar("obj_vely", obs["observation"][15], timestep)
            writer.add_scalar("obj_velz", obs["observation"][16], timestep)
            writer.add_scalar("obj_velrx", obs["observation"][17], timestep)
            writer.add_scalar("obj_velry", obs["observation"][18], timestep)
            writer.add_scalar("obj_velrz", obs["observation"][19], timestep)
            writer.add_scalar("grip_velx", obs["observation"][20], timestep)
            writer.add_scalar("grip_vely", obs["observation"][21], timestep)
            writer.add_scalar("grip_velz", obs["observation"][22], timestep)
            writer.add_scalar("grip_vel_1", obs["observation"][23], timestep)
            writer.add_scalar("grip_vel_2", obs["observation"][24], timestep)
        timestep_dict = {}
        for key in ["fc1", "fc2"]:
            acts_tensor = activations[key]
            acts_tensor = F.elu(acts_tensor)
            timestep_dict[key] = acts_tensor.cpu()
        acts_dict["grasp"][ep_num].append(timestep_dict)
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
        if timestep >= env._max_episode_steps: break
    output = choreographer(state_inp, hx, cx)
    state_value = output["state"]
    action_values = output["actions"]
    hx = output["hidden"]
    cx = output["cell"]
    prob = F.softmax(action_values, dim=-1)
    act_model = prob.max(-1, keepdim=True)[1].data
    action_out = act_model.to(torch.device("cpu"))
    goal = obs['desired_goal']
    object_pos = obs['observation'][3:6]
    while np.linalg.norm(goal - object_pos) >= 0.01 and timestep <= env._max_episode_steps:
        # env.render()
        action = [0, 0, 0, 0, 0]
        _, output = behaviour_net(state_inp)
        if ep_num == max_eps - 1:
            writer.add_scalar("grip_x", obs["observation"][0], timestep)
            writer.add_scalar("grip_y", obs["observation"][1], timestep)
            writer.add_scalar("grip_z", obs["observation"][2], timestep)
            writer.add_scalar("obj_x", obs["observation"][3], timestep)
            writer.add_scalar("obj_y", obs["observation"][4], timestep)
            writer.add_scalar("obj_z", obs["observation"][5], timestep)
            writer.add_scalar("obj_rel_grip_x", obs["observation"][6], timestep)
            writer.add_scalar("obj_rel_grip_y", obs["observation"][7], timestep)
            writer.add_scalar("obj_rel_grip_z", obs["observation"][8], timestep)
            writer.add_scalar("grip_state_1", obs["observation"][9], timestep)
            writer.add_scalar("grip_state_2", obs["observation"][10], timestep)
            writer.add_scalar("obj_rotx", obs["observation"][11], timestep)
            writer.add_scalar("obj_roty", obs["observation"][12], timestep)
            writer.add_scalar("obj_rotz", obs["observation"][13], timestep)
            writer.add_scalar("obj_velx", obs["observation"][14], timestep)
            writer.add_scalar("obj_vely", obs["observation"][15], timestep)
            writer.add_scalar("obj_velz", obs["observation"][16], timestep)
            writer.add_scalar("obj_velrx", obs["observation"][17], timestep)
            writer.add_scalar("obj_velry", obs["observation"][18], timestep)
            writer.add_scalar("obj_velrz", obs["observation"][19], timestep)
            writer.add_scalar("grip_velx", obs["observation"][20], timestep)
            writer.add_scalar("grip_vely", obs["observation"][21], timestep)
            writer.add_scalar("grip_velz", obs["observation"][22], timestep)
            writer.add_scalar("grip_vel_1", obs["observation"][23], timestep)
            writer.add_scalar("grip_vel_2", obs["observation"][24], timestep)
        timestep_dict = {}
        for key in ["fc1", "fc2"]:
            acts_tensor = activations[key]
            acts_tensor = F.elu(acts_tensor)
            timestep_dict[key] = acts_tensor.cpu()
        acts_dict["retract"][ep_num].append(timestep_dict)
        act_tensor = get_behaviour_from_model_output(output, action_out)

        for i in range(len(goal - object_pos)):
            action[i] = act_tensor[i].cpu().detach().numpy()
        
        action[3] = -0.01
        obs, reward, done, info = env.step(action)
        timestep += 1
        state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
        # noise = Normal(0, 0.1).sample(sample_shape=state_inp.size()).type(FloatTensor)
        # state_inp = state_inp + (args.noise / 100) * noise
        object_pos = obs['observation'][3:6]
        if timestep >= env._max_episode_steps: break

    while True: #limit the number of timesteps in the episode to a fixed duration
        # env.render()
        action = [0, 0, 0, 0, 0]
        action[3] = -0.01 # keep the gripper closed
        _, output = behaviour_net(state_inp)

        obs, reward, done, info = env.step(action)
        timestep += 1
        state_inp = torch.from_numpy(flattened_env.observation(obs)).type(FloatTensor)
        # noise = Normal(0, 0.1).sample(sample_shape=state_inp.size()).type(FloatTensor)
        # state_inp = state_inp + (args.noise / 100) * noise
        if timestep >= env._max_episode_steps: break
    ep_num += 1
    if info['is_success'] == 1.0:
        success +=1
        print("success")
    if ep_num % 100==0:            
        print("num episodes {}, success {}".format(ep_num, success))
        success = 0

torch.save(acts_dict, "acts.pt")
