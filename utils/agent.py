import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os 
import copy
import time 
import random
import math
import tqdm
import numpy as np 
from utils.utils import printOut, K_means
import matplotlib.pyplot as plt


class A2CAgent(object):
    
    def __init__(self, actor, critic, select_model, args, env, dataGen):
        self.actor = actor
        self.critic = critic
        self.select_model = select_model
        self.args = args
        self.device = args['device']
        self.env = env 
        self.dataGen = dataGen 
        out_file = open(os.path.join(args['log_dir'], 'results.txt'),'w+') 
        self.prt = printOut(out_file, args['stdout_print'])
        print("agent is initialized")

    def train(self, tb_logger):
        args = self.args 
        env = self.env
        dataGen = self.dataGen
        actor = self.actor
        critic = self.critic 
        prt = self.prt
        actor.train()
        critic.train()
        max_epochs = args['n_train']
        num_agents = args['num_agents']
        n_nodes = args['n_nodes']
        actor_optim = optim.Adam(actor.parameters(), lr=args['actor_net_lr'])
        critic_optim = optim.Adam(critic.parameters(), lr=args['critic_net_lr'])
        
        best_model = 1000
        val_model = 1000
        r_test = []
        r_val = []
        s_t = time.time()
        print("training started")
        for i in range(max_epochs):
            data = dataGen.get_train_next()
            env.input_data = data 
            state, avail_actions = env.reset()
            data = torch.from_numpy(data[:, :, :2].astype(np.float32)).to(self.device)
            # [b_s, hidden_dim, n_nodes]
            static_hidden = actor.emd_stat(data).permute(0, 2, 1)
            # critic inputs 
            static = torch.from_numpy(env.input_data[:, :, :2].astype(np.float32)).permute(0, 2, 1).to(self.device)
            w = torch.from_numpy(env.input_data[:, :, 2].reshape(env.batch_size, env.n_nodes, 1).astype(np.float32)).to(self.device)
            
            # lstm initial states 
            hx = torch.zeros(1, env.batch_size, args['hidden_dim']).to(self.device)
            cx = torch.zeros(1, env.batch_size, args['hidden_dim']).to(self.device)
            last_hh = (hx, cx)
       
            # prepare input
            # 这里的ter如果是terminate就需要修改！暂时还没有改
            ter = np.zeros(env.batch_size).astype(np.float32)
            decoder_input = static_hidden[:, :, env.n_nodes-1].unsqueeze(2)
       
            #[n_nodes, rem_time]
            time_vec_truck = np.zeros([env.batch_size, num_agents, 2])
            # [n_nodes, rem_time, weigth]
            time_vec_drone = np.zeros([env.batch_size, num_agents, 3])
            
            # storage containers 
            logs = []
            actions = []
            probs = []
            time_step = 0
            while time_step < args['decode_len']:
                terminated = torch.from_numpy(ter.astype(np.float32)).to(self.device)
                idx_truck_list = []
                idx_drone_list = []
                for k in range(num_agents):
                    for j in range(2):
                        # truck takes action
                        if j == 0:
                            avail_actions_truck = torch.from_numpy(avail_actions[:, :, k, 0].reshape([env.batch_size, env.n_nodes]).astype(np.float32)).to(self.device)
                            dynamic_truck = torch.from_numpy(np.expand_dims(state[:, :, k, 0], 2)).to(self.device)
                            idx_truck, prob, logp, last_hh = actor.forward(static_hidden, dynamic_truck, decoder_input, last_hh,
                                                                         terminated, avail_actions_truck)
                            b_s = np.where(np.logical_and(avail_actions[:, :, k, 1].sum(axis=1)>1, env.sortie[:, k]==0))[0]
                            avail_actions[b_s, idx_truck[b_s].cpu(), :, :] = 0             # 相当于mask操作
                            avail_actions_drone = torch.from_numpy(avail_actions[:, :, k, 1].reshape([env.batch_size, env.n_nodes]).astype(np.float32)).to(self.device)
                            idx_truck_list.append(idx_truck)
                            idx = idx_truck
                        else:
                            dynamic_drone = torch.from_numpy(np.expand_dims(state[:, :, k, 1], 2)).to(self.device)
                            idx_drone, prob, logp, last_hh = actor.forward(static_hidden, dynamic_drone, decoder_input, last_hh,
                                                                         terminated, avail_actions_drone)
                            idx = idx_drone
                            idx_drone_list.append(idx_drone)
                            b_s = np.where(avail_actions[:, :, k, 1].sum(axis=1)>1)[0]
                            avail_actions[b_s, idx_drone[b_s].cpu(), :, :] = 0

                        decoder_input = torch.gather(static_hidden, 2, idx.view(-1, 1, 1).expand(env.batch_size, args['hidden_dim'], 1)).detach()
                        logs.append(logp.unsqueeze(1))
                        actions.append(idx.unsqueeze(1))
                        probs.append(prob.unsqueeze(1))
               
                state, avail_actions, ter, time_vec_truck, time_vec_drone = env.step(idx_truck_list, idx_drone_list, time_vec_truck, time_vec_drone, ter)
                time_step += 1
            print("epochs: ", i)
            # self.draw(data, actions)
            # actions = torch.cat(actions, dim=1)  # (batch_size, seq_len)
            logs = torch.cat(logs, dim=1)  # (batch_size, seq_len)
            # Query the critic for an estimate of the reward
            critic_est = critic(static, w).view(-1)
            R = env.current_time.astype(np.float32) 
            R = torch.from_numpy(R).to(self.device)
            # 记录Reward
            tb_logger.log_value('reward', R.mean(), i)
            advantage = (R - critic_est)
            actor_loss = torch.mean(advantage.detach() * logs.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)
            # 记录loss
            tb_logger.log_value('actor_loss', actor_loss, i)
            tb_logger.log_value('critic_loss', critic_loss, i)
            actor_optim.zero_grad()
            actor_loss.backward()
            # 记录grad_norms
            # grad_norms = self.clip_grad_norms(actor.parameters(), args['max_grad_norm'])
            # grad_norms, grad_norms_clipped = grad_norms
            grad_norms = torch.nn.utils.clip_grad_norm_(actor.parameters(), args['max_grad_norm'])
            if not args['no_tensorboard']:
                tb_logger.log_value('grad_norm', grad_norms, i)
            actor_optim.step()
            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args['max_grad_norm'])
            critic_optim.step()
        
            e_t = time.time() - s_t
            print("e_t: ", e_t)
            if i % args['test_interval'] == 0:
                R = self.test()
                r_test.append(R)
                np.savetxt("trained_models/test_rewards.txt", r_test)
            
                print("testing average rewards: ", R)
                if R < best_model:
                 #   R_val = self.test(inference=False, val=False)
                    best_model = R
                    num = str(i // args['save_interval'])
                    torch.save(actor.state_dict(), 'trained_models/' + 'n' + str(n_nodes) + '/' + str(num_agents) + '_agents/best_model_actor_truck_params.pkl')
                    torch.save(critic.state_dict(), 'trained_models/' + 'n' + str(n_nodes) + '/' + str(num_agents) + '_agents/best_model_critic_params.pkl')
                   
            if i % args['save_interval'] ==0:
                num = str(i // args['save_interval'])
                torch.save(actor.state_dict(), 'trained_models/' + 'n' + str(n_nodes) + '/' + str(num_agents) + '_agents/' + num + '_actor_truck_params.pkl')
                torch.save(critic.state_dict(), 'trained_models/' + 'n' + str(n_nodes) + '/' + str(num_agents) + '_agents/' + num + '_critic_params.pkl')


    def train_distributed(self, tb_logger):
        args = self.args
        env = self.env
        dataGen = self.dataGen
        actor = self.actor
        critic = self.critic
        prt = self.prt
        for i in range(args['num_agents']):
            actor[i].train()
            critic[i].train()
        max_epochs = args['n_train']
        num_agents = args['num_agents']
        n_nodes = args['n_nodes']
        actor_optim = []
        critic_optim = []
        for i in range(args['num_agents']):
            actor_optim.append(optim.Adam(actor[i].parameters(), lr=args['actor_net_lr']))
            critic_optim.append(optim.Adam(critic[i].parameters(), lr=args['critic_net_lr']))

        best_model = 1000
        val_model = 1000
        r_test = []
        r_val = []
        s_t = time.time()
        print("Distributed training started")
        for i in range(max_epochs):
            data = dataGen.get_train_next()
            env.input_data = data
            state, avail_actions = env.reset()
            data = torch.from_numpy(data[:, :, :2].astype(np.float32)).to(self.device)
            # [b_s, hidden_dim, n_nodes]
            static_hidden = []
            for j in range(num_agents):
                static_hidden.append(actor[j].emd_stat(data).permute(0, 2, 1))
            # critic inputs
            static = torch.from_numpy(env.input_data[:, :, :2].astype(np.float32)).permute(0, 2, 1).to(self.device)
            w = [[torch.from_numpy(env.input_data[:, :, 2].reshape(env.batch_size, env.n_nodes, 1)
                                   .astype(np.float32)).to(self.device)] for i in range(num_agents)]
            w = torch.zeros(env.batch_size, env.n_nodes, 1, num_agents).to(self.device)
            row_indices = torch.arange(env.batch_size)
            # lstm initial states
            hx = [torch.zeros(1, env.batch_size, args['hidden_dim']).to(self.device)]
            cx = [torch.zeros(1, env.batch_size, args['hidden_dim']).to(self.device)]
            last_hh = (hx, cx)
            last_hh = [
                (torch.zeros(1, env.batch_size, args['hidden_dim']).to(self.device), torch.zeros(1, env.batch_size, args['hidden_dim']).to(self.device))
                for j in range(num_agents)
            ]

            # prepare input
            # 这里的ter如果是terminate就需要修改！暂时还没有改
            ter = np.zeros(env.batch_size).astype(np.float32)
            decoder_input = []
            for j in range(num_agents):
                decoder_input.append(static_hidden[j][:, :, env.n_nodes - 1].unsqueeze(2))

            # [n_nodes, rem_time]
            time_vec_truck = np.zeros([env.batch_size, num_agents, 2])
            # [n_nodes, rem_time, weigth]
            time_vec_drone = np.zeros([env.batch_size, num_agents, 3])

            # storage containers
            logs = [[] for j in range(num_agents)]
            actions = [[] for j in range(num_agents)]
            probs = []
            time_step = 0
            while time_step < args['decode_len']:
                terminated = torch.from_numpy(ter.astype(np.float32)).to(self.device)
                idx_truck_list = []
                idx_drone_list = []
                for k in range(num_agents):
                    for j in range(2):
                        # truck takes action
                        if j == 0:
                            avail_actions_truck = torch.from_numpy(
                                avail_actions[:, :, k, 0].reshape([env.batch_size, env.n_nodes]).astype(np.float32)).to(
                                self.device)
                            dynamic_truck = torch.from_numpy(np.expand_dims(state[:, :, k, 0], 2)).to(self.device)
                            idx_truck, prob, logp, last_hh[k] = actor[k].forward(static_hidden[k], dynamic_truck, decoder_input[k],
                                                                           last_hh[k],
                                                                           terminated, avail_actions_truck)
                            b_s = np.where(np.logical_and(avail_actions[:, :, k, 1].sum(axis=1) > 1, env.sortie[:, k] == 0))[
                                0]
                            avail_actions[b_s, idx_truck[b_s].cpu(), :, :] = 0  # 相当于mask操作
                            avail_actions_drone = torch.from_numpy(
                                avail_actions[:, :, k, 1].reshape([env.batch_size, env.n_nodes]).astype(np.float32)).to(
                                self.device)
                            idx_truck_list.append(idx_truck)
                            idx = idx_truck
                        else:
                            dynamic_drone = torch.from_numpy(np.expand_dims(state[:, :, k, 1], 2)).to(self.device)
                            idx_drone, prob, logp, last_hh[k] = actor[k].forward(static_hidden[k], dynamic_drone, decoder_input[k],
                                                                           last_hh[k],
                                                                           terminated, avail_actions_drone)
                            idx = idx_drone
                            idx_drone_list.append(idx_drone)
                            b_s = np.where(avail_actions[:, :, k, 1].sum(axis=1) > 1)[0]
                            avail_actions[b_s, idx_drone[b_s].cpu(), :, :] = 0

                        decoder_input[k] = torch.gather(static_hidden[k], 2,
                                                     idx.view(-1, 1, 1).expand(env.batch_size, args['hidden_dim'],
                                                                               1)).detach()
                        logs[k].append(logp.unsqueeze(1))
                        actions[k].append(idx.unsqueeze(1))
                        w[row_indices, idx, :, k] = 1
                        probs.append(prob.unsqueeze(1))
                state, avail_actions, ter, time_vec_truck, time_vec_drone = env.step(idx_truck_list, idx_drone_list,
                                                                                     time_vec_truck, time_vec_drone,
                                                                                     ter)
                time_step += 1
            print("epochs: ", i)
            w[:, n_nodes - 1, :, :] = 0
            # self.draw(data, actions)
            # actions = torch.cat(actions, dim=1)  # (batch_size, seq_len)
            R = env.current_time.astype(np.float32)
            R = torch.from_numpy(R).to(self.device)
            agent_R_min = env.agent_time.min(axis=1).astype(np.float32)
            agent_R_min = torch.from_numpy(agent_R_min).to(self.device)
            # 记录Reward
            tb_logger.log_value('reward', R.mean(), i)
            actor_loss_num = 0
            critic_loss_num = 0
            actor_loss = []
            critic_loss = []
            for j in range(num_agents):
                logs[j] = torch.cat(logs[j], dim=1)  # (batch_size, seq_len)
                # Query the critic for an estimate of the reward
                # agent_static = static * w[:, :, :, j]
                critic_est = critic[j](static, w[:, :, :, j], w, j).view(-1)
                agent_R = env.agent_time[:, j].astype(np.float32)
                agent_R = torch.from_numpy(agent_R).to(self.device)
                advantage = agent_R - critic_est
                actor_loss_num += (torch.mean(advantage.detach() * logs[j].sum(dim=1)))
                critic_loss_num += (torch.mean(advantage ** 2))
                actor_loss.append(torch.mean(advantage.detach() * logs[j].sum(dim=1)))
                critic_loss.append(torch.mean(advantage ** 2))
            # 记录loss
            tb_logger.log_value('actor_loss', actor_loss_num / num_agents, i)
            tb_logger.log_value('critic_loss', critic_loss_num / num_agents, i)
            for j in range(num_agents):
                actor_optim[j].zero_grad()
                actor_loss[j].backward()
                # 记录grad_norms
                actor_optim[j].step()
                torch.nn.utils.clip_grad_norm_(actor[j].parameters(), args['max_grad_norm'])
                critic_optim[j].zero_grad()
                critic_loss[j].backward()
                critic_optim[j].step()
                torch.nn.utils.clip_grad_norm_(critic[j].parameters(), args['max_grad_norm'])

            e_t = time.time() - s_t
            print("e_t: ", e_t)
            if i % args['test_interval'] == 0:
                R = self.test_distributed()
                r_test.append(R)
                np.savetxt("trained_models/test_rewards.txt", r_test)

                print("testing average rewards: ", R)
                # if R < best_model:
                #     #   R_val = self.test(inference=False, val=False)
                #     best_model = R
                #     num = str(i // args['save_interval'])
                #     torch.save(actor.state_dict(), 'trained_models/' + 'n' + str(n_nodes) + '/' + str(
                #         num_agents) + '_agents/best_model_actor_truck_params.pkl')
                #     torch.save(critic.state_dict(), 'trained_models/' + 'n' + str(n_nodes) + '/' + str(
                #         num_agents) + '_agents/best_model_critic_params.pkl')

            # if i % args['save_interval'] == 0:
            #     num = str(i // args['save_interval'])
            #     torch.save(actor.state_dict(), 'trained_models/' + 'n' + str(n_nodes) + '/' + str(
            #         num_agents) + '_agents/' + num + '_actor_truck_params.pkl')
            #     torch.save(critic.state_dict(), 'trained_models/' + 'n' + str(n_nodes) + '/' + str(
            #         num_agents) + '_agents/' + num + '_critic_params.pkl')

    def test(self):
        args = self.args 
        env = self.env 
        dataGen = self.dataGen
        actor  = self.actor
        n=2
        prt = self.prt 
        actor.eval()
        num_agents = args['num_agents']
        data = dataGen.get_test_all()
        env.input_data = data 
        state, avail_actions = env.reset()

        time_vec_truck = np.zeros([env.batch_size, num_agents, 2])
        time_vec_drone = np.zeros([env.batch_size, num_agents, 3])
        sols = []
        costs = []
        with torch.no_grad():
            data = torch.from_numpy(data[:, :, :2].astype(np.float32)).to(self.device)
            static_hidden = actor.emd_stat(data).permute(0, 2, 1)
            # lstm initial states 
            hx = torch.zeros(1, env.batch_size, args['hidden_dim']).to(self.device)
            cx = torch.zeros(1, env.batch_size, args['hidden_dim']).to(self.device)
            last_hh = (hx, cx)
       
            # prepare input 
            ter = np.zeros(env.batch_size).astype(np.float32)
            decoder_input = static_hidden[:, :, env.n_nodes-1].unsqueeze(2)
            time_step = 0
            while time_step < args['decode_len']:
                terminated = torch.from_numpy(ter.astype(np.float32)).to(self.device)
                idx_truck_list = []
                idx_drone_list = []
                for i in range(num_agents):
                    for j in range(2):
                        # truck takes action
                        if j == 0:
                            avail_actions_truck = torch.from_numpy(avail_actions[:, :, i, 0].reshape([env.batch_size, env.n_nodes]).astype(np.float32)).to(self.device)
                            dynamic_truck = torch.from_numpy(np.expand_dims(state[:, :, i, 0], 2)).to(self.device)
                            idx_truck, prob, logp, last_hh = actor.forward(static_hidden, dynamic_truck, decoder_input, last_hh,
                                                                         terminated, avail_actions_truck)
                            b_s = np.where(np.logical_and(avail_actions[:, :, i, 1].sum(axis=1)>1, env.sortie[:, i]==0))[0]
                            avail_actions[b_s, idx_truck[b_s].cpu(), :, :] = 0
                            avail_actions_drone = torch.from_numpy(avail_actions[:, :, i, 1].reshape([env.batch_size, env.n_nodes]).astype(np.float32)).to(self.device)
                            idx_truck_list.append(idx_truck)
                            idx = idx_truck
                        else:
                            dynamic_drone = torch.from_numpy(np.expand_dims(state[:, :, i, 1], 2)).to(self.device)
                            idx_drone, prob, logp, last_hh = actor.forward(static_hidden, dynamic_drone, decoder_input, last_hh,
                                                                         terminated, avail_actions_drone)
                            idx = idx_drone
                            idx_drone_list.append(idx_drone)
                            b_s = np.where(avail_actions[:, :, i, 1].sum(axis=1)>1)[0]
                            avail_actions[b_s, idx_drone[b_s].cpu(), :, :] = 0

                        decoder_input =  torch.gather(static_hidden, 2, idx.view(-1, 1, 1).expand(env.batch_size, args['hidden_dim'], 1)).detach()

                state, avail_actions, ter, time_vec_truck, time_vec_drone = env.step(idx_truck_list, idx_drone_list, time_vec_truck, time_vec_drone, ter)
                time_step += 1
                sols.append([idx_truck[n], idx_drone[n]])
                costs.append(env.time_step[n])
                
        R = copy.copy(env.current_time)
        costs.append(env.current_time[n])
        print("finished: ", sum(terminated).item())

        fname = 'test_results-{}-len-{}.txt'.format(args['test_size'], 
                                                               args['n_nodes'])
        fname = 'results/' + fname
        np.savetxt(fname, R)
        actor.train()
        return R.mean()

    def test_distributed(self):
        args = self.args
        env = self.env
        dataGen = self.dataGen
        actor = self.actor
        n = 2
        prt = self.prt
        for i in range(args['num_agents']):
            actor[i].eval()
        num_agents = args['num_agents']
        data = dataGen.get_test_all()
        env.input_data = data
        state, avail_actions = env.reset()

        time_vec_truck = np.zeros([env.batch_size, num_agents, 2])
        time_vec_drone = np.zeros([env.batch_size, num_agents, 3])
        sols = []
        costs = []
        with torch.no_grad():
            data = torch.from_numpy(data[:, :, :2].astype(np.float32)).to(self.device)
            static_hidden = []
            for i in range(num_agents):
                static_hidden.append(actor[i].emd_stat(data).permute(0, 2, 1))
            # lstm initial states
            hx = torch.zeros(1, env.batch_size, args['hidden_dim']).to(self.device)
            cx = torch.zeros(1, env.batch_size, args['hidden_dim']).to(self.device)
            last_hh = [(hx, cx) for i in range(num_agents)]

            # prepare input
            ter = np.zeros(env.batch_size).astype(np.float32)
            decoder_input = []
            for j in range(num_agents):
                decoder_input.append(static_hidden[j][:, :, env.n_nodes - 1].unsqueeze(2))
            time_step = 0
            while time_step < args['decode_len']:
                terminated = torch.from_numpy(ter.astype(np.float32)).to(self.device)
                idx_truck_list = []
                idx_drone_list = []
                for i in range(num_agents):
                    for j in range(2):
                        # truck takes action
                        if j == 0:
                            avail_actions_truck = torch.from_numpy(
                                avail_actions[:, :, i, 0].reshape([env.batch_size, env.n_nodes]).astype(np.float32)).to(
                                self.device)
                            dynamic_truck = torch.from_numpy(np.expand_dims(state[:, :, i, 0], 2)).to(self.device)
                            idx_truck, prob, logp, last_hh[i] = actor[i].forward(static_hidden[i], dynamic_truck, decoder_input[i],
                                                                           last_hh[i],
                                                                           terminated, avail_actions_truck)
                            b_s = np.where(np.logical_and(avail_actions[:, :, i, 1].sum(axis=1) > 1, env.sortie[:, i] == 0))[
                                0]
                            avail_actions[b_s, idx_truck[b_s].cpu(), :, :] = 0
                            avail_actions_drone = torch.from_numpy(
                                avail_actions[:, :, i, 1].reshape([env.batch_size, env.n_nodes]).astype(np.float32)).to(
                                self.device)
                            idx_truck_list.append(idx_truck)
                            idx = idx_truck
                        else:
                            dynamic_drone = torch.from_numpy(np.expand_dims(state[:, :, i, 1], 2)).to(self.device)
                            idx_drone, prob, logp, last_hh[i] = actor[i].forward(static_hidden[i], dynamic_drone, decoder_input[i],
                                                                           last_hh[i],
                                                                           terminated, avail_actions_drone)
                            idx = idx_drone
                            idx_drone_list.append(idx_drone)
                            b_s = np.where(avail_actions[:, :, i, 1].sum(axis=1) > 1)[0]
                            avail_actions[b_s, idx_drone[b_s].cpu(), :, :] = 0

                        decoder_input[i] = torch.gather(static_hidden[i], 2,
                                                     idx.view(-1, 1, 1).expand(env.batch_size, args['hidden_dim'],
                                                                               1)).detach()

                state, avail_actions, ter, time_vec_truck, time_vec_drone = env.step(idx_truck_list, idx_drone_list,
                                                                                     time_vec_truck, time_vec_drone,
                                                                                     ter)
                time_step += 1
                sols.append([idx_truck[n], idx_drone[n]])
                costs.append(env.time_step[n])

        R = copy.copy(env.current_time)
        costs.append(env.current_time[n])
        print("finished: ", sum(terminated).item())

        fname = 'test_results-{}-len-{}.txt'.format(args['test_size'],
                                                    args['n_nodes'])
        fname = 'results/' + fname
        np.savetxt(fname, R)
        for i in range(args['num_agents']):
            actor[i].train()
        return R.mean()


    def sampling_batch(self, sample_size):
        val_size = self.dataGen.get_test_all().shape[0]
        best_rewards = np.ones(sample_size)*100000
        best_sols = np.zeros([sample_size, self.args['decode_len'], 2])
        args = self.args 
        env = self.env 
        dataGen = self.dataGen
        actor  = self.actor
     
     
        actor.eval()
        actor.set_sample_mode(True)
        times = []
        initial_t = time.time()
        data = dataGen.get_test_all()
        data_list = [np.expand_dims(data[i, ...], axis=0) for i in range(data.shape[0])]
        best_rewards_list = []
        for d in data_list:
            data = np.repeat(d, sample_size, axis=0)
            env.input_data = data 
            
            state, avail_actions = env.reset()
            time_vec_truck = np.zeros([sample_size, 2])
            time_vec_drone = np.zeros([sample_size, 3])
            with torch.no_grad():
                data = torch.from_numpy(data[:, :, :2].astype(np.float32)).to(self.device)
                # [b_s, hidden_dim, n_nodes]
                static_hidden = actor.emd_stat(data).permute(0, 2, 1)
            
                # lstm initial states 
                hx = torch.zeros(1, sample_size, args['hidden_dim']).to(self.device)
                cx = torch.zeros(1, sample_size, args['hidden_dim']).to(self.device)
                last_hh = (hx, cx)
        
                # prepare input 
                ter = np.zeros(sample_size).astype(np.float32)
                decoder_input = static_hidden[:, :, env.n_nodes-1].unsqueeze(2)
                time_step = 0
                while time_step < args['decode_len']:
                    terminated = torch.from_numpy(ter).to(self.device)
                    idx_truck_list = []
                    idx_drone_list = []
                    for j in range(2):
                        # truck takes action 
                        if j == 0:
                            avail_actions_truck = torch.from_numpy(avail_actions[:, :, 0].reshape([sample_size, env.n_nodes]).astype(np.float32)).to(self.device)
                            dynamic_truck = torch.from_numpy(np.expand_dims(state[:, :, 0], 2)).to(self.device)
                            idx_truck, prob, logp, last_hh = actor.forward(static_hidden, dynamic_truck, decoder_input, last_hh, 
                                                                        terminated, avail_actions_truck)
                            b_s = np.where(np.logical_and(avail_actions[:, :, 1].sum(axis=1)>1, env.sortie==0))[0]
                            avail_actions[b_s, idx_truck[b_s].cpu(), 1] = 0
                            avail_actions_drone = torch.from_numpy(avail_actions[:, :, 1].reshape([sample_size, env.n_nodes]).astype(np.float32)).to(self.device)
                            idx = idx_truck 
                        
                        else:
                            dynamic_drone = torch.from_numpy(np.expand_dims(state[:, :, 1], 2)).to(self.device)
                            idx_drone, prob, logp, last_hh = actor.forward(static_hidden, dynamic_drone, decoder_input, last_hh, 
                                                                        terminated, avail_actions_drone)
                            idx = idx_drone 

                        decoder_input =  torch.gather(static_hidden, 2, idx.view(-1, 1, 1).expand(sample_size, args['hidden_dim'], 1)).detach()
                
                    state, avail_actions, ter, time_vec_truck, time_vec_drone = env.step(idx_truck.cpu().numpy(), idx_drone.cpu().numpy(), time_vec_truck, time_vec_drone, ter)
                    time_step += 1
            
            R = copy.copy(env.current_time)
            print('R.shape:', R.shape)
            best_rewards = R.min(axis=0)
            print('best_rewards:', best_rewards)
            t = time.time() - initial_t
            times.append(t)
            best_rewards_list.append(best_rewards)        

        os.makedirs(os.path.dirname(f'results/best_rewards_list_{sample_size}_samples.txt'), exist_ok=True)
        np.savetxt(f'results/best_rewards_list_{sample_size}_samples.txt', best_rewards_list)
        return best_rewards, times


    def test_draw(self):
        args = self.args
        env = self.env
        dataGen = self.dataGen
        actor = self.actor
        n = 2
        prt = self.prt
        actor.eval()
        num_agents = args['num_agents']
        data = dataGen.get_test_all()
        env.input_data = data
        state, avail_actions = env.reset()

        time_vec_truck = np.zeros([env.batch_size, num_agents, 2])
        time_vec_drone = np.zeros([env.batch_size, num_agents, 3])
        actions = []
        costs = []
        with torch.no_grad():
            data = torch.from_numpy(data[:, :, :2].astype(np.float32)).to(self.device)
            static_hidden = actor.emd_stat(data).permute(0, 2, 1)
            # lstm initial states
            hx = torch.zeros(1, env.batch_size, args['hidden_dim']).to(self.device)
            cx = torch.zeros(1, env.batch_size, args['hidden_dim']).to(self.device)
            last_hh = (hx, cx)

            # prepare input
            ter = np.zeros(env.batch_size).astype(np.float32)
            decoder_input = static_hidden[:, :, env.n_nodes - 1].unsqueeze(2)
            time_step = 0
            while time_step < args['decode_len']:
                terminated = torch.from_numpy(ter.astype(np.float32)).to(self.device)
                idx_truck_list = []
                idx_drone_list = []
                for i in range(num_agents):
                    for j in range(2):
                        # truck takes action
                        if j == 0:
                            avail_actions_truck = torch.from_numpy(
                                avail_actions[:, :, i, 0].reshape([env.batch_size, env.n_nodes]).astype(np.float32)).to(
                                self.device)
                            dynamic_truck = torch.from_numpy(np.expand_dims(state[:, :, i, 0], 2)).to(self.device)
                            idx_truck, prob, logp, last_hh = actor.forward(static_hidden, dynamic_truck, decoder_input,
                                                                           last_hh,
                                                                           terminated, avail_actions_truck)
                            b_s = np.where(np.logical_and(avail_actions[:, :, i, 1].sum(axis=1) > 1, env.sortie[:, i] == 0))[0]
                            avail_actions[b_s, idx_truck[b_s].cpu(), :, :] = 0
                            avail_actions_drone = torch.from_numpy(
                                avail_actions[:, :, i, 1].reshape([env.batch_size, env.n_nodes]).astype(np.float32)).to(
                                self.device)
                            idx_truck_list.append(idx_truck)
                            idx = idx_truck
                        else:
                            dynamic_drone = torch.from_numpy(np.expand_dims(state[:, :, i, 1], 2)).to(self.device)
                            idx_drone, prob, logp, last_hh = actor.forward(static_hidden, dynamic_drone, decoder_input,
                                                                           last_hh,
                                                                           terminated, avail_actions_drone)
                            idx = idx_drone
                            idx_drone_list.append(idx_drone)
                            b_s = np.where(avail_actions[:, :, i, 1].sum(axis=1) > 1)[0]
                            avail_actions[b_s, idx_drone[b_s].cpu(), :, :] = 0

                        decoder_input = torch.gather(static_hidden, 2,
                                                     idx.view(-1, 1, 1).expand(env.batch_size, args['hidden_dim'],
                                                                               1)).detach()
                        actions.append(idx.unsqueeze(1))
                state, avail_actions, ter, time_vec_truck, time_vec_drone = env.step(idx_truck_list, idx_drone_list,
                                                                                     time_vec_truck, time_vec_drone,
                                                                                     ter)
                time_step += 1
                # sols.append([idx_truck[n], idx_drone[n]])
                costs.append(env.time_step[n])

        self.draw(data, actions, num_agents)
        R = copy.copy(env.current_time)
        costs.append(env.current_time[n])
        print("finished: ", sum(terminated).item())

        fname = 'test_results-{}-len-{}.txt'.format(args['test_size'],
                                                    args['n_nodes'])
        fname = 'results/' + fname
        np.savetxt(fname, R)
        actor.train()
        return R.mean()

    def draw(self, data, actions, num_agents):
        # 画出车辆和无人机的轨迹(前3个)
        n_nodes = data.shape[1]
        idx = 0
        for graph in data[:3]:
            # 先画出节点
            graph = graph.cpu().numpy()
            x = [point[0] for point in graph]
            y = [point[1] for point in graph]
            plt.scatter(x, y)
            i = 0
            for x1, y1 in zip(x, y):
                plt.annotate('%s' % i, xy=(x1, y1), xytext=(5, -10), textcoords='offset points')
                i += 1
            # 取每个agent的动作
            truck1_action = [n_nodes - 1]
            drone1_action = [n_nodes - 1]
            truck2_action = [n_nodes - 1]
            drone2_action = [n_nodes - 1]
            for i in range(0, len(actions), 4):
                truck1_action.append(actions[i][idx])
                drone1_action.append(actions[i + 1][idx])
                truck2_action.append(actions[i + 2][idx])
                drone2_action.append(actions[i + 3][idx])
            self.draw_arrow(graph, drone1_action, color='red', line_style='--')
            self.draw_arrow(graph, drone2_action, color='skyblue', line_style='--')
            self.draw_arrow(graph, truck1_action, color='darkred', line_style='solid')
            self.draw_arrow(graph, truck2_action, color='blue', line_style='solid')
            idx += 1
            plt.show()

    def draw_arrow(self, graph, actions, color, line_style):
        for i in range(len(actions) - 1):
            if actions[i] == actions[i+1]:
                continue
            else:
                plt.arrow(graph[actions[i]][0], graph[actions[i]][1],
                          graph[actions[i + 1]][0] - graph[actions[i]][0],
                          graph[actions[i + 1]][1] - graph[actions[i]][1],
                          color=color, linestyle=line_style, head_width=2, head_length=2)
                print(actions[i], actions[i+1])

            # plt.show()
