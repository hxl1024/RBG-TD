import numpy as np 
import os 
import torch 
import random
from utils.options import ParseParams
from utils.env_no_comb import Env, DataGenerator
from model.nnets import Actor, Critic, CriticNetwork
from model.SelectModel import SelectModel
from utils.a2cagent import A2CAgent
from utils.region import RegionAndSolution
import time
from tensorboard_logger import Logger as TbLogger


if __name__ == '__main__':
    args = ParseParams()   
    random_seed = args['random_seed']
    if random_seed is not None and random_seed > 0:
        print("# Set random seed to %d!!" % random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    max_epochs = args['n_train']
    args['device'] = torch.device("cuda:0") if torch.cuda.is_available else torch.device("cpu")
    # save_path = args['save_path']
    n_nodes = args['n_nodes']
    dataGen = DataGenerator(args)
    data = dataGen.get_train_next()
    data = dataGen.get_test_all()
    env = Env(args, data)
    actor = []
    critic = []
    if args['train_mode'] == "'distributed'":
        for i in range(args['num_agents']):
            actor.append(Actor(args['hidden_dim']))
            critic.append(CriticNetwork(
                2,
                args['embedding_dim'],
                args['hidden_dim'],
                args['rnn_layers'],
                args['num_agents'],
                ))
    else:
        actor = Actor(args['hidden_dim'])
        critic = Critic(args['hidden_dim'])
    select_model = SelectModel(args)
    save_path = args['save_path'] + 'n' + str(n_nodes) + '/' + str(args['num_agents']) + '_agents/' + 'distributed/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(args['model_dir']):
        os.makedirs(args['model_dir'])
    else:
        path = save_path + '/best_model_actor_truck_params.pkl'
        if os.path.exists(path):
            path = save_path + '/best_model_actor_truck_params.pkl'
            path_critic = save_path + '/best_model_critic_params.pkl'
            if args['train_mode'] == "'distributed'":
                for i in range(args['num_agents']):
                    actor[i].load_state_dict(torch.load(path, map_location='cpu'))
                    critic[i].load_state_dict(torch.load(path_critic, map_location='cpu'))
            else:
                actor.load_state_dict(torch.load(path, map_location='cpu'))
                critic.load_state_dict(torch.load(path_critic, map_location='cpu'))

            print("Succesfully loaded keys")
    
    agent = A2CAgent(actor, critic, select_model, args, env, dataGen)
    regionSolver = RegionAndSolution(agent, select_model, args, env, dataGen)
    if args['train']:
        # Optionally configure tensorboard
        tb_logger = None
        if not args['no_tensorboard']:
            tb_logger = TbLogger(
                os.path.join(args['log_dir'], "n{}_{}agents".format(args['n_nodes'], args['num_agents']),
                             args['run_name']))
        # Start train
        if args['train_mode'] == "'distributed'":
            agent.train_distributed(tb_logger)
        else:
            regionSolver.train(tb_logger)
    else:
        if args['sampling']:
            best_R = agent.sampling_batch(args['n_samples'])
        else:
            R = agent.test_draw()
            print(R)