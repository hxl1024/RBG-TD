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

    def merge_regions(self, regions2, cov, merged_region, sampling=True):
        # 选取merge的region（cost最大的）
        mean = self.mean_regions_points(regions2)  # (num_region,3)
        dist_square = np.sum((mean[:, None, :2] - mean[None, :, :2]) ** 2, -1)
        arg_near = np.argsort(dist_square, -1)[:, 1:self.near_K + 1]
        cov_k = cov[np.arange(len(regions2))[:, None], arg_near][merged_region]  # (regions2_num,near_K)
        cov_k = torch.nn.functional.softmax(cov_k, -1)
        # print(merged_region, cov_k)
        if sampling:
            selected = torch.multinomial(cov_k, 1)  # (regions2_num, )
        else:
            selected = torch.argmax(cov_k, 1)
        logits = torch.log(cov_k[selected[0].item()])
        return arg_near[merged_region][selected].item(), logits


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