# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
from __future__ import print_function

import json
import math
import os
import sys
import time
from datetime import datetime
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x

print_grad = True


class printOut(object):
    def __init__(self, f=None, stdout_print=True):
        ''' 
        This class is used for controlling the printing. It will write in a 
        file f and screen simultanously.
        '''
        self.out_file = f
        self.stdout_print = stdout_print

    def print_out(self, s, new_line=True):
        """Similar to print but with support to flush and output to a file."""
        if isinstance(s, bytes):
            s = s.decode("utf-8")

        if self.out_file:
            self.out_file.write(s)
            if new_line:
                self.out_file.write("\n")
        self.out_file.flush()

        # stdout
        if self.stdout_print:
            print(s, end="", file=sys.stdout)
            if new_line:
                sys.stdout.write("\n")
            sys.stdout.flush()

    def print_time(self, s, start_time):
        """Take a start time, print elapsed duration, and return a new time."""
        self.print_out("%s, time %ds, %s." % (s, (time.time() - start_time) + "  " + str(time.ctime())))
        return time.time()


def get_time():
    '''returns formatted current time'''
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def K_means(data, K, beta, max_iterations=20, plot=False, device='cpu'):
    data = torch.tensor(data).to(device)
    loc = torch.tensor(data[:, :2]).to(device)
    loc_center = torch.rand(K, 2).to(device)
    for n in range(max_iterations):
        loc2_center = torch.zeros((K, 2)).to(device)
        num_center = torch.zeros(K).to(device)
        track_node = []  # points for each center, None for zero; shape=[size1*2,size2*2,...,sizeK*2]
        #         track_demand=[]  #shape: size1*n
        if n == max_iterations - 1:
            for i in range(K):
                track_node.append([])
        #             track_demand.append([])

        dis1 = torch.argmin(distance(loc[:, None, :], loc_center, beta, device=device), 1)  # (N,)
        for i, j in enumerate(dis1):
            loc2_center[j] = loc2_center[j] + loc[i]
            num_center[j] = num_center[j] + 1
            if n == max_iterations - 1:
                track_node[j].append(data[i, :])

        for k in range(K):
            if num_center[k] > 0:
                loc_center[k] = loc2_center[k] / num_center[k]
                if n == max_iterations - 1:
                    track_node[k] = torch.stack(track_node[k], 0)  # (num,3)
            #                 track_demand[k]=torch.stack(track_demand[k],0)
            else:
                loc_center[k] = torch.rand(2)
                if n == max_iterations - 1:
                    track_node[k] = None
    #                     track_demand[k]=None

    if plot:
        for k in range(K):
            plt.plot(track_node[k].numpy()[:, 0], track_node[k].numpy()[:, 1], '.')
        plt.plot(loc_center[:, 0].numpy(), loc_center[:, 1].numpy(), 'bo')
        plt.show()

    return track_node  # [(num_in_cluster, 3)]*K


def distance(x, y, beta=0, depot_loc=torch.tensor([0.5, 0.5]), eps1=0.00001, device='cpu'):
    #     print(x.shape,y.shape)
    depot_loc = depot_loc.to(device)
    dis = torch.sum((x - y) ** 2, -1).to(device)
    if beta != 0:
        a1 = torch.sum((x - depot_loc) * (y - depot_loc), -1)
        b1 = (torch.sqrt(torch.sum((x - depot_loc) ** 2, -1) + eps1))
        c1 = (torch.sqrt(torch.sum((y - depot_loc) ** 2, -1) + eps1))
        ang = torch.acos(a1 / b1 / c1)
    else:
        ang = 0
    return dis + beta * ang


def stacks(li, device=None, dim=0, stack_dim=0, min_length=None, constant=0):
    m = max([s.shape[dim] for s in li])
    if min_length is not None:
        m = max(m, min_length)
    li1 = []
    pad_dim = [0] * len(li[0].shape) * 2
    size_n = torch.tensor([s.shape[dim] for s in li])
    for s in li:
        pad_dim[len(s.shape) * 2 - 2 * dim - 1] = m - s.shape[dim]
        li1.append(F.pad(s.to(device), pad_dim, value=constant))
    #         print(li1[-1])
    #         print(li1[-1].shape)
    return torch.stack(li1, stack_dim), size_n


def rotate(x, Q, depot):
    return torch.matmul(Q[:, None, :, :], (x[:, :, :, None] - depot[:, None, :, None]))[:, :, :, 0] + depot[:, None, :]


def rotate_matrix(theta, device=torch.device('cpu')):
    cos1 = torch.cos(theta)  # (batch,)
    sin1 = torch.sin(theta)
    Q = torch.stack([torch.stack([cos1, sin1], -1), torch.stack([-sin1, cos1], -1)], 1)
    return Q.to(device)


def pi2regions(regions, actions):
    sub_regions_n = len(actions)
    regions1 = []
    region_idx= []
    for i, r0 in enumerate(regions):
        r1 = []
        idx = []
        for action in actions:
            # pi是一个batch的action，shape=[batch,K]
            pi = torch.cat([tensor.unsqueeze(1) for tensor in action], dim=1).squeeze(-1).cpu()
            pre_j = -1
            # 因为1组由一个truck和drone构成，二者跳步
            pre_pre_j = -1
            for k, j in enumerate(pi[i]):
                if j == 0:
                    break
                elif pre_pre_j == -1 and pre_j == -1:
                    pre_pre_j = pre_j
                    pre_j = j
                    r1.append([r0[j]])
                else:
                    if pre_pre_j == j or pre_j == j:
                        pre_pre_j = pre_j
                        pre_j = j
                        continue
                    pre_pre_j = pre_j
                    pre_j = j
                    r1[-1].append(r0[j])
                    idx.append(k)
        r1 = [torch.stack(route, 0) for route in r1]
        region_idx.append(idx)
        regions1.append(r1)
    return regions1


def divide_region(regions1):
    regions2 = []
    for r1 in regions1:  # [(route_len,3)]*route_num
        route_mean = torch.stack([torch.mean(route, 0)[:2] for route in r1], 0)  # (route_num,2)
        route_cnt = [route.shape[0] for route in r1]  # (route_num,)
        route_total_cnt = sum(route_cnt)
        route_mean_mean = torch.mean(route_mean, 0)  # (2,)
        U, S, V = torch.pca_lowrank((route_mean - route_mean_mean))
        route_PCA = torch.matmul((route_mean - route_mean_mean), V[:, :1])[:, 0]  # (route_num, )
        sorted_route_PCA = torch.argsort(route_PCA)
        sub1 = []
        sub2 = []
        cnt = 0
        allinsub1 = False
        for i, idx in enumerate(sorted_route_PCA):
            if cnt > route_total_cnt / 2:
                break
            cnt += route_cnt[idx]
            sub1.append(r1[idx])
            if i == len(sorted_route_PCA) - 1:
                allinsub1 = True
        if not allinsub1:
            for idx in sorted_route_PCA[i:]:
                sub2.append(r1[idx])
        regions2.append(sub1)
        regions2.append(sub2)
    return regions2


def parse_idx(a, idx, dim=0):
    output = []
    for i in range(len(idx) - 1):
        output.append(a[idx[i]:idx[i + 1]])
    return output


def divide_and_update_region(regions1, merged_regions2, should_update):
    regions2 = []
    for cnt1, r1 in enumerate(regions1):  # [(route_len,3)]*route_num
        if should_update[cnt1]:
            route_mean = torch.stack([torch.mean(route, 0)[:2] for route in r1], 0)  # (route_num,2)
            #         print('route_mean=',route_mean.shape)
            route_cnt = [route.shape[0] for route in r1]  # (route_num,)
            route_total_cnt = sum(route_cnt)
            route_mean_mean = torch.mean(route_mean, 0)  # (2,)
            U, S, V = torch.pca_lowrank((route_mean - route_mean_mean))
            route_PCA = torch.matmul((route_mean - route_mean_mean), V[:, :1])[:, 0]  # (route_num, )
            #         print('route_PCA.shape=',route_PCA.shape,route_PCA)
            sorted_route_PCA = torch.argsort(route_PCA)
            #         print('sorted_route_PCA=',sorted_route_PCA)
            sub1 = []
            sub2 = []
            cnt = 0
            for i, idx in enumerate(sorted_route_PCA):
                if cnt > route_total_cnt / 2:
                    break
                cnt += route_cnt[idx]
                sub1.append(r1[idx])

            for idx in sorted_route_PCA[i:]:
                sub2.append(r1[idx])
        else:  # worse than previous solution, therefore not updating
            sub1 = merged_regions2[cnt1 * 2]
            sub2 = merged_regions2[cnt1 * 2 + 1]

        regions2.append(sub1)
        regions2.append(sub2)
    #     print('len=',len(regions2))
    return regions2


def costs_of_regions(regions1, depot):
    costs = torch.zeros(len(regions1))
    for i, region in enumerate(regions1):
        for route in region:
            costs[i] = costs[i] + cost_of_route(route, depot)
    #     print(costs)
    return costs


def cost_of_route(route, depot):
    c = torch.sum(torch.sqrt(torch.sum((route[1:, :2] - route[:-1, :2]) ** 2, 1)), 0)
    return torch.sqrt(torch.sum((route[0, :2] - depot) ** 2)) + c + torch.sqrt(torch.sum((route[-1, :2] - depot) ** 2))