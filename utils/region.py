import os
import time
import math
from tqdm import tqdm
from torch.nn import DataParallel
import torch
import numpy as np
from utils.utils import K_means, pi2regions, stacks, rotate_matrix, rotate, divide_region, parse_idx, divide_and_update_region, costs_of_regions
import torch.optim as optim

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

class RegionAndSolution(object):
    def __init__(self, a2cAgent, select_model, args, env, dataGen):
        self.a2cAgent = a2cAgent
        self.select_model = select_model
        self.args = args
        self.env = env
        self.dataGen = dataGen

    def train(self, tb_logger=None, running_cost=0):
        args = self.args
        env = self.env
        K = args['K']
        dataGen = self.dataGen
        a2cAgent = self.a2cAgent
        select_model = self.select_model
        a2cAgent.actor.train()
        a2cAgent.critic.train()
        max_epochs = args['n_train']
        num_agents = args['num_agents']
        n_nodes = args['n_nodes']
        # actor_optim = optim.Adam(a2cAgent.actor.parameters(), lr=args['actor_net_lr'])
        critic_optim = optim.Adam(a2cAgent.critic.parameters(), lr=args['critic_net_lr'])
        if args['train_selection_only']:
            optimizer = torch.optim.SGD([{'params': select_model.parameters(), 'lr': args['lr2']}])
        else:
            optimizer = torch.optim.SGD([{'params': a2cAgent.actor.parameters(), 'lr': args['lr1']}] +
                                        [{'params': select_model.parameters(), 'lr': args['lr2']}])
        print("training started")
        for i in range(max_epochs):
            # 每个epoch训练
            loc_data, depot_data = dataGen.get_train_next()
            batch_size = len(loc_data)
            # 使用K-means算法初始化region
            init_regions = [K_means(node, K, args['beta'], args['Kmeans_iter']) for node in loc_data]
            # 记录每个sample的region id范围
            idx_regions = [i * K for i in range(batch_size + 1)]
            regions = [[] for i in range(batch_size)]
            remain_regions = [[] for i in range(batch_size)]
            remain_costs = torch.zeros(batch_size)
            for step in tqdm(range(args['iter_steps'])):
                # 根据region id范围，给每个region的subregion加上depot（这里能不能不用for呢）
                for j in range(len(idx_regions) - 1):
                    for k in range(idx_regions[j], idx_regions[j + 1]):
                        init_regions[j][k - idx_regions[j]] = torch.cat(
                            (torch.from_numpy(depot_data[j]), init_regions[j][k - idx_regions[j]]), dim=0)
                regions0 = []
                # regions0是region集合（list）
                for region in init_regions:
                    regions0 = regions0 + region
                if args['train_selection_only']:
                    costs1, ll, pi = self.merge_judge(regions0, args['device'], cal_ll=True, args=args)
                else:
                    R, loss1, critic_loss, actions, critic_est = self.merge_train(regions0, args['device'], i, step, idx_regions, cal_ll=True, args=args, tb_logger=tb_logger)

                # 将region根据actions分成两部分
                part_regions = pi2regions(regions0, actions)
                costs1 = R
                if step == 0:
                    regions2 = divide_region(part_regions)
                    costss1 = parse_idx(costs1, idx_regions, dim=0)  # [(num_regions1)]*multiple
                else:
                    should_update = costs1 < merged_costs1 * (1 + args['up_rate_train'])  # shape=(num_regions1,)
                    assert len(merged_regions2) == len(part_regions) * 2, 'mismatch of len of regions'
                    regions2 = divide_and_update_region(part_regions, merged_regions2, should_update)

                    costss1 = parse_idx(torch.where(should_update, costs1, merged_costs1), idx_regions,
                                        dim=0)  # [(num_regions1)]*multiple

                if step >= 1:
                    if args['enable_running_cost']:
                        loss2 = ((costs1 - merged_costs1 - running_cost) * logits2).mean()
                    else:
                        loss2 = ((costs1 - merged_costs1) * logits2).mean()
                    running_cost = running_cost * args['running_cost_alpha'] + (1 - args['running_cost_alpha']) * (
                            costs1 - merged_costs1).mean().item()
                    # # print('running_cost=', running_cost, 'costs1=', costs1.mean().item(), ', merged_costs1=',
                    #       merged_costs1.mean().item())
                else:
                    loss2 = torch.tensor(0)

                optimizer.zero_grad()

                if args['train_selection_only']:
                    if step >= 1:
                        loss2.backward()
                else:
                    loss = loss1 + loss2
                    loss.backward()

                with open('log_{}.txt'.format(args['save_model']), 'a') as f:
                    f.write(
                        'step={}, loss1={}, loss2={}, running_cost={}\n'.format(step, loss1.item(), loss2.item(),
                                                                                running_cost))

                if args['enable_gradient_clipping']:
                    grad_norms = self.clip_grad_norms(optimizer.param_groups, args['max_grad_norm'])
                    # print(grad_norms)
                optimizer.step()

                # costs = torch.tensor([c.sum().item() for c in costss1]) + remain_costs

                for j in range(batch_size):
                    regions[j] = remain_regions[j] + regions2[idx_regions[j] * 2:idx_regions[j + 1] * 2]  # [[(_route_len,3)]*route_num_in_a_region]*num_region, 50 points
                #             plot_regions1(regions2,divide_num=2)

                merged_costs1 = []  # [(region_cnt,)]*multiple
                logits2 = []  # [(region_cnt,)]*multiple
                merged_regions2 = []  # [[(route_len,3)]*route_cnt]*(region_num*multiple)
                depot = [depot_data[i, 0, :2] for i in range(batch_size)]
                depot = [torch.from_numpy(arr).view(2) for arr in depot]
                for j in range(batch_size):
                    costs2 = costs_of_regions(regions[j], depot[j])  # (2K,)
                    cov = select_model(regions[j])
                    regions1, regions2, selected, logits2_i, merged_costs1_i, merged_regions2_i = select_model.merge_regions(
                        regions[j], cov, costs2)  # selected.shape=(2K,)
                    logits2.append(logits2_i)
                    merged_costs1.append(merged_costs1_i)
                    merged_regions2 = merged_regions2 + merged_regions2_i
                    idx_regions[j + 1] = idx_regions[j] + len(regions1)
                    remain_regions[j] = regions2
                    remain_costs[j] = torch.sum(costs2) - torch.sum(costs2 * selected)
                    regions0 = [torch.cat(r, 0) for r in regions1]  # [(cnt,3)]*merged_num
                    init_regions[j] = regions0
                logits2 = torch.cat(logits2)
                merged_costs1 = torch.cat(merged_costs1).to(args['device'])
                print('epoch{}, step{}:{}'.format(i, step, R.mean()))
                # print('data{},step{}:{}'.format(batch_id, step, costs))
                if step == args['iter_steps'] - 1:
                    # 记录loss
                    tb_logger.log_value('actor_loss', loss1, i)
                    tb_logger.log_value('critic_loss', critic_loss, i)
                    tb_logger.log_value('loss1 & loss2', loss, i)
                    tb_logger.log_value('costs', R.mean(), i)
            if (args['checkpoint_epochs'] != 0 and i % args['checkpoint_epochs'] == 0) or i == max_epochs - 1:
                print('Saving model and state...')
                torch.save(
                    {
                        'SelectModel': get_inner_model(select_model).state_dict(),
                        'AttentionModel': get_inner_model(a2cAgent.actor).state_dict(),
                        'CriticModel': get_inner_model(a2cAgent.critic).state_dict(),
                    },
                    os.path.join(args['model_dir'], 'epoch-{}.pt'.format(i))
                )

    def merge_judge(self, regions, device, cal_ll=False, args=None):
        # 每个region的node，因为数目不一致，统一为最大数目，多出来的都是0；size_n就是batch * K（region的数目）
        nodes, size_n = stacks(regions, device, constant=0)  # (K,num_in_cluster,3)
        #     demands,_ =stacks(demands,device)
        batch = {}
        batch['depot'] = nodes[0].to(device)
        batch['loc'] = nodes[1:, :, :2].to(device)
        batch['demand'] = nodes[1:, :, 2].to(device)
        # 这段的作用是将loc旋转一定的角度，这个角度是随机的，这样可以增加数据的多样性，从而提高模型的泛化能力
        if args.enable_random_rotate_eval:
            theta = torch.rand(nodes.shape[0]) * 2 * 3.1415926535
            Q = rotate_matrix(theta, device)
            batch['loc'] = rotate(batch['loc'], Q, batch['depot'])
            # print('loc.shape=', batch['loc'].shape)

        t1 = time.time()
        with torch.no_grad():
            cost1, ll, pi1 = self.a2cAgent.actor.model((batch, size_n.to(device)), return_pi=True, cal_ll=cal_ll)
        t2 = time.time()

        return cost1, ll, pi1

    def merge_train(self, regions0, device, epoch, step, idx_regions, cal_ll=True, args=None, tb_logger=None):
        nodes, size_n = stacks(regions0, device, constant=0)  # (K,num_in_cluster,3)
        nodes = nodes.float()
        #     demands,_ =stacks(demands,device)
        batch = {}
        batch['depot'] = nodes[0].to(device)
        batch['loc'] = nodes[1:, :, :2].to(device)
        # batch['demand'] = nodes[1:, :, 2].to(device)

        # if args.enable_random_rotate_train:
        #     theta = torch.rand(nodes.shape[0]) * 2 * 3.1415926535
        #     Q = rotate_matrix(theta, device)
        #     batch['loc'] = rotate(batch['loc'], Q, batch['depot'])
        #     # print('loc.shape=', batch['loc'].shape)

        R, loss1, critic_loss, actions, critic_est = self.a2cAgent.train(nodes, size_n.to(device), epoch, step, idx_regions, return_pi=True, cal_ll=cal_ll, tb_logger=tb_logger)
        return R, loss1, critic_loss, actions, critic_est

    def clip_grad_norms(self, param_groups, max_norm=math.inf):
        """
        Clips the norms for all param groups to max_norm and returns gradient norms before clipping
        :param optimizer:
        :param max_norm:
        :param gradient_norms_log:
        :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
        """
        grad_norms = [
            torch.nn.utils.clip_grad_norm_(
                group['params'],
                max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
                norm_type=2
            )
            for group in param_groups
        ]
        grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
        return grad_norms, grad_norms_clipped