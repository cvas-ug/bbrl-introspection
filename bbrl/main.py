import os
import gym
import argparse
import torch
import json
import numpy as np
import torch.multiprocessing as mp

from shared_adam import SharedAdam
from models import BehaviourNetwork, ChoreographNetwork

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = '1'

    parser = argparse.ArgumentParser(description="A3C")
    parser.add_argument("command", metavar="<command>",
                        help="[approach|manipulate|retract|choreograph]")
    parser.add_argument('--use-cuda',default=True,
                    help='run on gpu.')
    parser.add_argument('--max-grad-norm', type=float, default=250,
                        help='value loss coefficient (default: 50)')
    parser.add_argument('--max-eps', type=float, default=10000,
                        help='max number of episodes (default: 10000)')
    parser.add_argument('--max-steps', type=float, default=50,
                        help='max number of steps per episode (default: 50)')
    parser.add_argument('--num-processes', type=int, default=2,
                        help='how many training processes to use (default: 4)')
    parser.add_argument('--save-interval', type=int, default=50,
                        help='model save interval (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.9,
                    help='discount factor for rewards (default: 0.9)')
    parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
    parser.add_argument("--noise", type=int)
    parser.add_argument("--weights-path", type=str)
    args = parser.parse_args()

    if args.command == "approach":
        from approach import train, test
    elif args.command == "grasp":
        from grasp import train, test
    elif args.command == "retract":
        from retract import train, test
    elif args.command == "choreograph":
        from choreograph import train, test
        
    multi_proc = mp.get_context('spawn')
    env = gym.make("FetchPickAndPlace-v1")
    shared_model = BehaviourNetwork(args.weights_path, args.command)
    if args.command == "choreograph":
        shared_model = ChoreographNetwork(args.weights_path, internal_states=True)
    if args.use_cuda:
        shared_model.cuda()
    
    torch.cuda.manual_seed_all(12)

    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()

    processes = []

    counter = multi_proc.Value('i', 0)
    lock = multi_proc.Lock()
    p = multi_proc.Process(target=test, args=(args.num_processes, args, shared_model, counter))

    p.start()
    processes.append(p)

    num_procs = args.num_processes
    
    if args.num_processes > 1:
        num_procs = args.num_processes - 1 

    for rank in range(0, num_procs):
        p = multi_proc.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()