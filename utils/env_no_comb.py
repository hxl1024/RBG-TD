import numpy as np
import os
import warnings
import collections
import copy
import time
from functools import reduce

def create_test_dataset(
        args):
    rnd = np.random.RandomState(seed=args['random_seed'])
    n_problems = args['test_size']
    n_nodes = args['n_nodes']
    data_dir = args['data_dir']
    # build task name and datafiles
    task_name = 'DroneTruck-size-{}-len-{}.txt'.format(n_problems, n_nodes)
    fname = os.path.join(data_dir, task_name)

    # create/load data
    if os.path.exists(fname):
        print('Loading dataset for {}...'.format(task_name))
        data = np.loadtxt(fname, delimiter=' ')
        data = data.reshape(-1, n_nodes, 3)
        input_data = data
    else:
        print('Creating dataset for {}...'.format(task_name))
        # Generate a training set of size n_problems on grid 50x50
        input_pnt = np.random.uniform(1, 100,
                                      size=(args['test_size'], args['n_nodes'] - 1, 2))
        input_pnt = np.concatenate([input_pnt, np.random.uniform(50, 51, size=(args['test_size'], 1, 2))], axis=1)
        demand = np.ones([args['test_size'], args['n_nodes'] - 1, 1])

        # make the last node depot 
        network = np.concatenate([demand, np.zeros([args['test_size'], 1, 1])], 1)
        input_data = np.concatenate([input_pnt, network], 2)
        np.savetxt(fname, input_data.reshape(-1, n_nodes * 3))

    return input_data


class DataGenerator(object):
    def __init__(self,
                 args):
        self.args = args
        self.rnd = np.random.RandomState(seed=args['random_seed'])
        # create test data
        self.test_data = create_test_dataset(args)

    def get_train_next(self):
        args = self.args
        input_pnt = np.random.uniform(1, 100,
                                      size=(args['test_size'], args['n_nodes'] - 1, 2))
        # input_pnt = np.concatenate([input_pnt, np.random.uniform(50, 51, size=(args['test_size'], 1, 2))], axis=1)
        demand = np.ones([args['batch_size'], args['n_nodes'] - 1, 1])
        depot = np.random.uniform(50, 51, size=(args['batch_size'], 1, 2))
        depot_demand = np.zeros([args['batch_size'], 1, 1])
        input_data = np.concatenate([input_pnt, demand], 2)
        depot_data = np.concatenate([depot, depot_demand], 2)

        return input_data.astype(np.float32), depot_data.astype(np.float32)

    def get_test_all(self):
        '''
        Get all test problems
        '''
        return self.test_data


class Env(object):
    def __init__(self, args, data):

        self.args = args
        self.rnd = np.random.RandomState(seed=args['random_seed'])
        self.input_data = data
        self.n_nodes = args['n_nodes']
        self.v_t = args['v_t']
        self.v_d = args['v_d']
        self.batch_size = args['batch_size']
        self.num_agents = args['num_agents']
        print("Using Not revisiting nodes")

    def reset(self, size_n = None):
        self.size_n = size_n
        self.batch_size = self.input_data[:, :, :2].shape[0]
        self.input_pnt = self.input_data[:, :, :2]
        self.n_nodes = self.input_data.shape[1]
        self.dist_mat = np.zeros([self.batch_size, self.n_nodes, self.n_nodes])
        for i in range(self.n_nodes):
            # 将距离矩阵计算出来，并存储存到dist_mat中
            for j in range(i + 1, self.n_nodes):
                self.dist_mat[:, i, j] = ((self.input_pnt[:, i, 0] - self.input_pnt[:, j, 0]) ** 2 + (
                            self.input_pnt[:, i, 1] - self.input_pnt[:, j, 1]) ** 2) ** 0.5
                self.dist_mat[:, j, i] = self.dist_mat[:, i, j]
        avail_actions = np.ones([self.batch_size, self.n_nodes, self.num_agents, 2], dtype=np.float32)
        avail_actions[:, 0, :, :] = np.zeros([self.batch_size, self.num_agents, 2])
        for i in range(self.input_data.shape[0]):
            avail_actions[i, size_n[i]:, :, :] = 0
        self.drone_mat = self.dist_mat / self.v_d
        self.state = np.ones([self.batch_size, self.n_nodes])
        self.state[:, 0] = np.zeros([self.batch_size])
        self.sortie = np.zeros([self.batch_size, self.num_agents])
        self.returned = np.ones([self.batch_size, self.num_agents])
        self.current_time = np.zeros(self.batch_size)
        self.agent_time = np.zeros([self.batch_size, self.num_agents])
        self.truck_loc = np.ones([self.batch_size, self.num_agents], dtype=np.int32) * (self.n_nodes - 1)
        self.drone_loc = np.ones([self.batch_size, self.num_agents], dtype=np.int32) * (self.n_nodes - 1)

        dynamic = np.zeros([self.batch_size, self.n_nodes, self.num_agents, 2], dtype=np.float32)
        for i in range(self.num_agents):
            dynamic[:, :, i, 0] = self.dist_mat[:, 0]
            dynamic[:, :, i, 1] = self.drone_mat[:, 0]
        return dynamic, avail_actions

    def step(self, idx_truck, idx_drone, time_vec_truck, time_vec_drone, terminated):
        old_sortie = copy.copy(self.sortie)
        old_truck_loc = copy.copy(self.truck_loc)
        # compute which action occurs first
        t_truck = []
        t_drone = []
        time_step = []
        for i in range(self.num_agents):
            t_truck.append(
                self.dist_mat[np.arange(self.batch_size, dtype=np.int64), self.truck_loc[:, i], idx_truck[i].cpu()])
            t_drone.append(
                self.drone_mat[np.arange(self.batch_size, dtype=np.int64), self.drone_loc[:, i], idx_drone[i].cpu()])
            # only count nonzero time movements: if trucks/drones stay at the same place, update based on other actions
            A = t_truck[i] + np.equal(t_truck[i], np.zeros(self.batch_size)).astype(int) * np.ones(
                self.batch_size) * 10000
            A = np.squeeze(A)  # 将A降维成1维
            B = t_drone[i] + np.equal(t_drone[i], np.zeros(self.batch_size)).astype(int) * np.ones(
                self.batch_size) * 10000
            B = np.squeeze(B)  # 将B降维成1维
            C = time_vec_truck[:, i, 1] + np.equal(time_vec_truck[:, i, 1], np.zeros(self.batch_size)).astype(
                int) * np.ones(self.batch_size) * 10000
            D = time_vec_drone[:, i, 1] + np.equal(time_vec_drone[:, i, 1], np.zeros(self.batch_size)).astype(
                int) * np.ones(self.batch_size) * 10000
            # time_step取A, B, C, D中的最小值
            time_step.append(np.minimum.reduce([A, B, C, D]))

        time_step = np.min(time_step, axis=0)
        # 如果全部到了 time_step自然为0，没有action
        b_s = np.where(terminated == 1)[0]
        time_step[b_s] = np.zeros(len(b_s))

        self.time_step = time_step
        # 确定目前的time_step
        self.current_time += time_step
        # 初始化mask矩阵（全零，后面才确定到底什么能选）
        avail_actions = np.zeros([self.batch_size, self.n_nodes, self.num_agents, 2], dtype=np.float32)
        dynamic = np.zeros([self.batch_size, self.n_nodes, self.num_agents, 2], dtype=np.float32)
        terminated = np.ones(self.batch_size, dtype=int)
        agent_terminated = np.zeros([self.batch_size, self.num_agents], dtype=int)
        for i in range(self.num_agents):
            # time_vec_truck大于0说明truck行驶到下一个node的时间比drone长
            time_vec_truck[:, i, 1] += np.logical_and(np.equal(time_vec_truck[:, i, 1], np.zeros(self.batch_size)),
                                                   np.greater(t_truck[i], np.zeros(self.batch_size))).astype(int) * (
                                                t_truck[i] - time_step) - \
                                    np.greater(time_vec_truck[:, i, 1], np.zeros(self.batch_size)) * (time_step)
            # time_vec_drone大于0说明drone行驶到下一个node的时间比truck长
            time_vec_drone[:, i, 1] += np.logical_and(np.equal(time_vec_drone[:, i, 1], np.zeros(self.batch_size)),
                                                   np.greater(t_drone[i], np.zeros(self.batch_size))).astype(int) * (
                                                t_drone[i] - time_step) - \
                                    np.greater(time_vec_drone[:, i, 1], np.zeros(self.batch_size)) * (time_step)

            # 如果time_vec_truck大于0，则truck_loc不改变；如果比drone能先到（即等于0），则更改为idx_truck
            self.truck_loc[:, i] += np.equal(time_vec_truck[:, i, 1], np.zeros(self.batch_size)) * (idx_truck[i].cpu() - self.truck_loc[:, i]).numpy()
            self.drone_loc[:, i] += np.equal(time_vec_drone[:, i, 1], np.zeros(self.batch_size)) * (idx_drone[i].cpu() - self.drone_loc[:, i]).numpy()

            # 记录到达下一个node需要的时间，0维的值是node的idx
            time_vec_truck[:, i, 0] = np.logical_and(np.less(time_step, t_truck[i]),
                                                     np.greater(time_vec_truck[:, i, 1], np.zeros(self.batch_size))) * idx_truck[i].cpu().numpy()
            time_vec_drone[:, i, 0] = np.logical_and(np.less(time_step, t_drone[i]),
                                                  np.greater(time_vec_drone[:, i, 1], np.zeros(self.batch_size))) * idx_drone[i].cpu().numpy()

            # update demand because of turck and drone
            # b_s是truck通过这次action能够抵达的instance索引
            b_s = np.where(np.equal(time_vec_truck[:, i, 1], np.zeros(self.batch_size)))[0]
            b_s = np.where(~np.equal(time_vec_truck[:, i, 0], np.ones(self.batch_size) * (self.n_nodes - 1)))[0]
            self.state[b_s, idx_truck[i][b_s].cpu()] = np.zeros(len(b_s))
            # drone到了的地方
            idx_satis = np.where(np.less(self.sortie[:, i] - np.equal(time_vec_drone[:, i, 1], 0), np.zeros(self.batch_size)))[0]
            self.state[idx_satis, idx_drone[i].cpu()[idx_satis]] -= np.equal(time_vec_drone[idx_satis, i, 1],
                                                                             np.zeros(len(idx_satis))) * self.state[idx_satis, idx_drone[i].cpu()[idx_satis]]
            # update sortie if drone served customer
            self.sortie[:, i][idx_satis] = np.ones(len(idx_satis))
            # 判断truck和drone是否会合了
            a = np.equal((self.truck_loc[:, i] == self.drone_loc[:, i]).astype(int) + (time_vec_drone[:, i, 1] == 0).astype(int) + (
                        time_vec_truck[:, i, 1] == 0).astype(int), 3)
            idx_stais = np.where(np.expand_dims(a, 1))[0]
            self.sortie[:, i][idx_stais] = np.zeros(len(idx_stais))
            self.returned[:, i] = np.ones(self.batch_size) - np.equal(
                (old_sortie[:, i] == 1).astype(int) + (self.sortie[:, i] == 1).astype(int) + (time_vec_drone[:, i, 1] == 0).astype(int), 3)
            self.returned[:, i][idx_stais] = np.ones(len(idx_stais))
            #######################################################################################
            # masking scheme
            #######################################################################################
            # 没有完成行动的vehicle让它保持该行进轨迹
            # for unfinished actions of truck: make only unfinished actions available
            b_s = np.where(np.expand_dims(time_vec_truck[:, i, 1], 1) > 0)[0]
            idx_fixed = time_vec_truck[b_s, i, np.zeros(len(b_s), dtype=np.int64)]
            avail_actions[b_s, idx_fixed.astype(int), i, 0] = np.ones(len(b_s))
            # for unfinished actions of drone: make only unfinished actions available
            b_s_d = np.where(np.expand_dims(time_vec_drone[:, i, 1], 1) > 0)[0]
            idx_fixed_d = time_vec_drone[b_s_d, i, np.zeros(len(b_s_d), dtype=np.int64)]
            avail_actions[b_s_d, idx_fixed_d.astype(int), i, 1] = np.ones(len(b_s_d))


            # 这一块还是有点点不对，到Line 214  # 3.25改了这里！！
            # otherwise, select any node with unsatisfied demand regardless sortie value
            # 这里我表示怀疑，不是greater_equal而是equal？
            # 如果drone已经送达，drone其他所有节点都是available的（暂时ok）
            a = np.equal(
                np.greater_equal(time_vec_truck[:, i, 1], 0).astype(int) + np.equal(time_vec_drone[:, i, 1], 0).astype(int) + np.equal(old_sortie[:, i], 0).astype(int)
                + (~np.equal(self.truck_loc[:, i], self.drone_loc[:, i])).astype(int), 4)
            # a = np.equal(
            #     np.greater_equal(time_vec_truck[:, i, 1], 0).astype(int) + np.equal(time_vec_drone[:, i, 1], 0).astype(
            #         int) + np.equal(old_sortie[:, i], 0).astype(int), 3)
            b_s = np.where(np.expand_dims(a, 1))[0]
            avail_actions[b_s, time_vec_truck[b_s, i, 0].astype(int), i, 1] = 1
            # 如果是drone回到truck了，那就可以是随便没访问的节点
            a = np.equal(
                np.greater_equal(time_vec_truck[:, i, 1], 0).astype(int) + np.equal(time_vec_drone[:, i, 1], 0).astype(
                    int) + np.equal(old_sortie[:, i], 1).astype(int), 3)
            b_s = np.where(np.expand_dims(a, 1))[0]
            avail_actions[b_s, :, i, 1] = np.greater(self.state[b_s, :], 0)

            # if drone has already selected returning node make it stay there
            # 如果drone选择了returning node就让它呆在那里等truck到来
            a = np.equal(np.equal(self.returned[:, i], 0).astype(int) + np.equal(time_vec_drone[:, i, 1], 0).astype(int), 2)
            b_s = np.where(np.expand_dims(a, 1))[0]
            avail_actions[b_s, :, i, 1] = 0
            avail_actions[b_s, self.drone_loc[b_s, i], i, 1] = 1

            # for truck that finished action select any node with customer demand
            # 如果货车结束行动了，所有没到达的地方都是available的（这段ok）
            b_s = np.where(np.expand_dims(time_vec_truck[:, i, 1], 1) == 0)[0]
            avail_actions[b_s, :, i, 0] += np.greater(self.state[b_s, :], 0)

            # if there is expected visit by drone to that customer node with sortie =0
            # don't make that node available to truck
            # 如果这个地方已经在被drone访问并且接它的truck到点了（说明不去drone送货地方集合），他就不能被truck和其他drone访问
            a = np.equal(np.equal(self.sortie[:, i], 0).astype(int) + np.greater(time_vec_drone[:, i, 1], 0).astype(int) + np.equal(
                time_vec_truck[:, i, 1], 0).astype(int), 3)
            b_s_s = np.where(np.expand_dims(a, 1))[0]
            idx_fixed_d = time_vec_drone[b_s_s, i, np.zeros(len(b_s_s), dtype=np.int64)]

            # make current location available if there is expected visit by drone
            # 如果drone要去truck的地方，并且drone还在路上且truck已经到了（何必呢？），则让他还能够被访问
            a = np.equal(
                np.equal(self.truck_loc[:, i], time_vec_drone[:, i, 0]).astype(int) + np.greater(time_vec_drone[:, i, 1], 0).astype(
                    int) + np.equal(time_vec_truck[:, i, 1], 0).astype(int), 3)
            b_s = np.where(np.expand_dims(a, 1))[0]
            avail_actions[b_s, self.truck_loc[:, i][b_s], i, 0] = 1

            # make the current location of drone available to truck if its stuck there
            # 如果无人机返回了，且已经到了，且货车已经到了，让无人机的位置能够被访问
            a = np.equal(np.equal(self.returned[:, i], 0).astype(int) + np.equal(time_vec_drone[:, i, 1], 0).astype(int) + np.equal(
                time_vec_truck[:, i, 1], 0).astype(int), 3)
            b_s = np.where(np.expand_dims(a, 1))[0]
            avail_actions[b_s, self.drone_loc[:, i][b_s], i, 0] = 1

            # 其他组到的地方不能走
            for j in range(self.num_agents):
                if j != i:
                    b_s = np.arange(self.batch_size).astype(int)
                    avail_actions[b_s, idx_truck[j].cpu(), i, :] = 0
                    avail_actions[b_s, idx_drone[j].cpu(), i, :] = 0

            # if the last customer left and both drone and truck at the same location
            # let the drone serve the last customer
            a = np.equal(np.equal(self.state.sum(axis=1), 1).astype(int) +
                         np.equal((avail_actions[:, :, i, 0] == avail_actions[:, :, i, 1]).sum(axis=1), self.n_nodes).astype(
                             int) +
                         np.equal(self.drone_loc[:, i], self.truck_loc[:, i]).astype(int), 3)
            b_s = np.where(a)[0]
            avail_actions[b_s, :, i, 0] = np.zeros(self.n_nodes)

            # if the last customer left and truck is scheduled to visit there, let drone fly to depot
            # 如果只剩一个customer了并且truck要去，让drone飞到depot
            a = np.equal(np.equal(self.state.sum(axis=1), 1).astype(int) + np.equal(time_vec_drone[:, i, 1], 0).astype(
                int) + np.greater(time_vec_truck[:, i, 1], 0).astype(int) + np.equal(self.returned[:, i], 1).astype(int), 4)
            b_s = np.where(a)[0]
            avail_actions[b_s, :, i, 1] = np.zeros(self.n_nodes)
            # open depot for drone and truck if there is no other options
            # 超过范围的全赋值0
            for j in range(self.input_data.shape[0]):
                avail_actions[j, self.size_n[j]:, :, :] = 0
            # 如果无人机和货车无处可去，则开放depot允许其返回
            avail_actions[:, 0, i, 0] += np.equal(avail_actions[:, :, i, 0].sum(axis=1), 0)
            avail_actions[:, 0, i, 1] += np.equal(avail_actions[:, :, i, 1].sum(axis=1), 0)


            dynamic[:, :, i, 0] = self.dist_mat[np.arange(self.batch_size), self.truck_loc[:, i]]
            dynamic[:, :, i, 1] = self.drone_mat[np.arange(self.batch_size), self.drone_loc[:, i]]


            terminated = np.logical_and(terminated, np.logical_and(np.equal(self.truck_loc[:, i], 0),
                                        np.equal(self.drone_loc[:, i], 0)).astype(int))

            b_s = np.where(np.logical_and(np.equal(self.truck_loc[:, i], 0),
                                        np.equal(self.drone_loc[:, i], 0)).astype(int) == 1)[0]
            b_s = np.where(reduce(np.logical_and, [np.equal(self.truck_loc[:, i], 0),
                                                   np.equal(self.drone_loc[:, i], 0),
                                                   ~np.equal(old_truck_loc[:, i], self.truck_loc[:, i])]))[0]
            self.agent_time[b_s, i] = self.current_time[b_s]
            agent_terminated[:, i] = np.logical_and(agent_terminated[:, i], np.logical_and(np.equal(self.truck_loc[:, i], 0),
                                        np.equal(self.drone_loc[:, i], 0)).astype(int))

        return dynamic, avail_actions, terminated, time_vec_truck, time_vec_drone
