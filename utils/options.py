import argparse
import time
import os
def str2bool(v):
    return v.lower() in ('true', '1')


def ParseParams():
    parser = argparse.ArgumentParser(description="TSP with Drone")
    # Data generation for Training and Testing 
    parser.add_argument('--n_nodes', default=20, type=int, help="Number of nodes")
    parser.add_argument('--R', default = 150, type=int, help="Drone battery life in time units")
    parser.add_argument('--v_t', default = 1, type=int, help="Speed of truck in m/s")
    parser.add_argument('--v_d', default = 2, type=int, help="Speed of drone in m/s")
    parser.add_argument('--max_w', default = 2.5, type=float, help="Max weight a drone can carry")
    parser.add_argument('--batch_size', default= 100,type=int, help='Batch size for training')
    parser.add_argument('--n_train', default=5000,type=int, help='# of episodes for training')
    parser.add_argument('--test_size', default=100,type=int, help='# of instances for testing')
    parser.add_argument('--data_dir', type=str, default='data1')
    parser.add_argument('--save_model', default='model1', type=str)
    parser.add_argument('--model_dir', default='pretrained', type=str)
    # parser.add_argument('--log_dir', default='logs/', help='Directory to write TensorBoard information to')
    parser.add_argument('--save_path', type=str, default='trained_models/')
    parser.add_argument('--test_interval', default=20,type=int, help='test every test_interval steps')
    parser.add_argument('--save_interval', default=1000,type=int, help='save every save_interval steps')
    parser.add_argument('--log_dir', default='logs',type=str, help='folder for saving prints')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--stdout_print', default=True, type=str2bool, help='print control')
    parser.add_argument('--num_agents', default=2, type=int, help='number of agents(1 agent for a truck and a drone)')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    # Neural Network Structure 
    
    # Embedding 
    parser.add_argument('--embedding_dim', default=128,type=int, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', default=128,type=int, help='Dimension of hidden layers in Enc/Dec')
    
    # Decoder: LSTM 
    parser.add_argument('--rnn_layers', default=1, type=int, help='Number of LSTM layers in the encoder and decoder')
    parser.add_argument('--forget_bias', default=1.0,type=float, help="Forget bias for BasicLSTMCell.")
    parser.add_argument('--dropout', default=0.1, type=float, help='The dropout prob')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    # Attention 
    parser.add_argument('--use_tanh', type=str2bool, default=False, help='use tahn before computing probs in attention')
    parser.add_argument('--mask_logits', type=str2bool, default=True, help='mask unavailble nodes probs')
    
    # Training
    parser.add_argument('--train', default=True, type=str2bool, help="whether to do the training or not")
    parser.add_argument('--actor_net_lr', default=1e-4,type=float, help="Set the learning rate for the actor network")
    parser.add_argument('--critic_net_lr', default=1e-4,type=float, help="Set the learning rate for the critic network")
    parser.add_argument('--random_seed', default= 5,type=int, help='')
    parser.add_argument('--max_grad_norm', default=2.0, type=float, help='Gradient clipping')
    parser.add_argument('--decode_len', default=30, type=int, help='Max number of steps per episode')
    parser.add_argument('--train_mode', default=None, type=str, help='The mode of epochs to train')
    parser.add_argument('--up_rate_train', default=0.01, type=float)
    parser.add_argument('--up_rate_eval', default=0.005, type=float)
    parser.add_argument('--checkpoint_epochs', default=5, type=int, help='The epoch to save the model')
    # Evaluation
    parser.add_argument('--sampling', default=True,type=str2bool, help="whether to do the batch sampling or not")
    parser.add_argument('--n_samples', default=5, type=int, help='the number of samples for batch sampling')
    # regions——K-means
    parser.add_argument('--near_K', default=2, type=int, help='the number of nearest neighbors to consider')
    parser.add_argument('--K', default=5, type=int, help='the number of output regions for initial K-means')
    parser.add_argument('--Kmeans_iter', default=20, type=int, help='the number of iterations for initial K-means')
    parser.add_argument('--beta', default=0.1, type=float, help='the parameter for distance')
    # regions--DRL
    parser.add_argument('--iter_steps', default=32, type=int, help='the number of iterations for DRL')
    parser.add_argument('--train_selection_only', default=False, type=str2bool, help='whether to train the selection model only')
    parser.add_argument('--enable_running_cost', default=True, action='store_true')
    parser.add_argument('--running_cost_alpha', default=0.99, type=float)
    parser.add_argument('--enable_gradient_clipping', action='store_true', default=True)
    parser.add_argument('--lr1', default=3e-5, type=float, help='random seed.')
    parser.add_argument('--lr2', default=1e-3, type=float, help='random seed.')
    args, unknown = parser.parse_known_args()
    args = vars(args)
    args['decode_len'] = max(round(args['n_nodes'] * 0.45), args['decode_len'])
    args['run_name'] = "{}_{}".format(args['run_name'], time.strftime("%Y%m%dT%H%M%S"))
    args['model_dir'] = os.path.join(
        args['model_dir'],
        "{}_{}".format(args['n_nodes'], args['num_agents']),
        args['run_name']
    )
    for key, value in sorted(args.items()):
        print("{}: {}".format(key,value))
    
    return args 