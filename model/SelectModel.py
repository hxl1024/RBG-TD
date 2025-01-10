import torch
import numpy as np
import torch.nn as nn
import random

class SelectModel(nn.Module):
    def __init__(self, args=None, embed_dim0=100, lstm_dim=100, embed_dims1=[1000, 100], device=torch.device('cuda:0')):
        super(SelectModel, self).__init__()
        self.embed_dim0 = embed_dim0
        self.lstm_dim = lstm_dim
        self.embed_dims1 = embed_dims1
        self.embed_dim = embed_dims1[-1]
        self.linear0 = nn.Linear(3, embed_dim0).to(device)
        self.LSTMCell = nn.LSTMCell(embed_dim0, lstm_dim).to(device)
        self.linears1 = []
        d0 = self.lstm_dim
        for d in self.embed_dims1[:-1]:
            self.linears1.append(nn.Linear(d0, d))
            self.linears1.append(nn.ReLU(inplace=True))
            d0 = d
        self.linears1.append(nn.Linear(d0, self.embed_dims1[-1]))
        self.linears1 = nn.Sequential(*self.linears1).to(device)
        self.h0 = nn.Parameter(torch.rand(1, lstm_dim, requires_grad=True)).to(device)
        self.c0 = nn.Parameter(torch.rand(1, lstm_dim, requires_grad=True)).to(device)
        self.near_K = args['near_K']
        self.args = args
        self.device = device

    def embed_route(self, route):  # route.shape=(route_cnt,3)
        # print(route.device)
        route_emb = self.linear0(route)
        for i, x in enumerate(route_emb):
            hx, cx = self.LSTMCell(x[None, :], (self.h0, self.c0))
        hx = self.linears1(hx)
        return hx  # shape=(1,emb_dims1[-1])

    def forward(self, regions2):
        region_emb = []
        for region in regions2:
            hxs = []
            for route in region:
                hxs.append(self.embed_route(route.to(self.device)))
            hxs = torch.sum(torch.cat(hxs, 0), 0, keepdim=True)  # (1,emb)
            region_emb.append(hxs)
        region_emb = torch.cat(region_emb, 0)  # (regions2_num,emb)
        region_emb = region_emb / torch.norm(region_emb, dim=-1, keepdim=True)  # divide by norm
        cov_matrix = torch.matmul(region_emb, region_emb.transpose(0, 1))  # (regions2_num,regions2_num)
        idx = torch.arange(len(regions2))
        cov_matrix[idx, idx] = -np.inf
        return cov_matrix  # ()

    def merge_regions(self, regions2, cov, costs2, sampling=True):
        near_K = self.near_K
        if near_K is None:
            near_K = len(regions2) - 1
        near_K = min(near_K, len(regions2) - 1)
        selected = np.zeros(cov.shape[0])
        regions1 = []
        merged_regions2 = []
        costs1 = []
        logits = []

        mean = self.mean_regions_points(regions2)  # (num_region,3)
        dist_square = np.sum((mean[:, None, :2] - mean[None, :, :2]) ** 2, -1)
        #         print('dist_square.shape=',dist_square.shape)
        arg_near = np.argsort(dist_square, -1)[:, 1:(near_K + 1)]  # (regions2_num,near_K)
        cov_k = cov[np.arange(len(regions2))[:, None], arg_near]  # (regions2_num,near_K)
        cov_k = torch.nn.functional.softmax(cov_k, -1)
        # cov_matrix[idx,idx]=0
        # print('cov2=',cov)
        if sampling:
            s = torch.multinomial(cov_k, 5)  # (regions2_num, )
        else:
            s = torch.argmax(cov_k, -1)
        s1 = s  # (regions2_num, )
        # s = arg_near[np.arange(len(regions2)), s.cpu()]  # (regions2_num, )
        s = [a_n[s_] for a_n, s_ in zip(arg_near, s.cpu())]

        ranges = list(range(cov_k.shape[0]))
        random.shuffle(ranges)
        for i in ranges:
            if selected[i]:
                continue
            has_topk = False
            # 找near_K中未被选中的
            for j in range(near_K):
                if not selected[s[i][j].item()]:
                    has_topk = True
                    break
            # pass if not the topk of this region all selected
            if not has_topk:
                continue
            selected[i] = 1
            selected[s[i][j].item()] = 1
            regions1.append(regions2[i] + regions2[s[i][j].item()])
            merged_regions2.append(regions2[i])
            merged_regions2.append(regions2[s[i][j].item()])
            costs1.append(costs2[i] + costs2[s[i][j].item()])
            logits.append(torch.log(cov_k[i, s1[i][j].item()]))
        regions3 = []
        pre_re = -1
        for i in range(cov.shape[0]):
            if not selected[i]:
                if pre_re == -1:
                    pre_re = i
                else:
                    regions1.append(regions2[pre_re] + regions2[i])
                    costs1.append(costs2[pre_re] + costs2[i])
                    logits.append(torch.log(cov[pre_re, i]))
                    merged_regions2.append(regions2[pre_re])
                    merged_regions2.append(regions2[i])
                    pre_re = -1
                selected[i] = 1
        return regions1, regions3, selected, torch.stack(logits), torch.tensor(costs1), merged_regions2


    def mean_regions_points(self, regions1):
        mean = np.zeros((len(regions1), 3))
        for i, region in enumerate(regions1):
            c = 0
            num = 0
            for route in region:
                c += torch.sum(route, 0).numpy()
                num += route.shape[0]
            mean[i, :] = c / num
        return mean