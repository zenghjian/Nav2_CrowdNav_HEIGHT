import argparse
import torch

# output_dir
def get_args():
    parser = argparse.ArgumentParser(description='RL')
    # the saving directory for train.py
    parser.add_argument(
        '--output_dir', type=str, default= 'data/pasrl') # 'VAEdata/test'
    parser.add_argument(
        '--lr', type=float, default=1e-5, help='learning rate (default: 7e-4)') 
    parser.add_argument(
        '--seed', type=int, default=0, help='random seed (default: 1)')
    parser.add_argument(
        '--weights', type=str)
    
    # resume training from an existing checkpoint or not
    parser.add_argument(
        '--resume', default=False, action='store_true')    
    # if resume = True, load from the following checkpoint
    parser.add_argument(
        '--load-path', default=None,
        help='path of weights for resume training')
    parser.add_argument(
        '--overwrite',
        default=False,
        action='store_true',
        help = "whether to overwrite the output directory in training")
    
    parser.add_argument(
        '--num_threads',
        type=int,
        default=1,
        help="number of threads used for intraop parallelism on CPU")    
    # only implement in testing
    parser.add_argument(
        '--phase', type=str, default='test')
    
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=12,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=2, 
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=30, 
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=5,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.0,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=15e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=400,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=True,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=20,
        help='log interval, one log per n updates (default: 10)')

    # RNN size
    parser.add_argument('--rnn_hidden_size', type=int, default=128, 
                        help='Size of Human Node RNN hidden state')
    parser.add_argument('--seq_length', type=int, default=30, 
                        help='Sequence length')
    parser.add_argument('--rnn_input_size', type=int, default=64,
                        help='Dimension of the node features')
    parser.add_argument('--rnn_output_size', type=int, default=128, 
                        help='Dimension of the node output') 

    parser.add_argument(
        '--env-name',
        default='CrowdSimDict-v0',
        help='name of the environment')


    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']

    return args