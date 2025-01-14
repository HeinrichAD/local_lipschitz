import os
import time
import numpy as np
import torch
from tqdm import tqdm

import my_config
import local_lipschitz.network_bound as network_bound
import local_lipschitz.utils as utils

import networks.tiny as exp
#import networks.compnet as exp
#import networks.mnist as exp
#import networks.cifar10 as exp
#import networks.alexnet as exp
#import networks.vgg16 as exp

def relu(x):
    return (x>0)*x

# setup
device = my_config.device

# which bounds to compute?
compute_rand =   1
compute_grad =   1
compute_global = 1
compute_local =  1

# get network and nominal input
main_dir = exp.main_dir
net = exp.net()
net = net.to(device)
net.eval()
x0 = exp.x0

# save files
rand_npz = os.path.join(main_dir, 'rand.npz')
grad_npz = os.path.join(main_dir, 'grad.npz')
global_npz = os.path.join(main_dir, 'global.npz')
local_npz = os.path.join(main_dir, 'local.npz')

# epsilons
n_bound = 30
eps = np.linspace(exp.eps_min, exp.eps_max, n_bound)

# evaluate entire network
x0.requires_grad = False
outputs = net(x0)

# iterate over each layer and get the output of each function
layers = net.layers
n_layers = len(layers)
X = [x0]
for i in range(n_layers):
    f = layers[i]
    X.append(f(X[-1]))

# check if layer-by-layer result is correct
print('layer-by-layer v. one-shot error:', torch.norm(X[-1] - outputs).item())


###############################################################################
# lower bound, random
if compute_rand:
    t0 = time.time()
    print('\nLOWER LIPSCHITZ BOUNDS, RANDOM')
    #lb = utils.lower_bound_random_many(net, x0, eps_lb_net, batch_size=exp.batch_size_lb)
    bound = utils.lower_bound_random_many(net, x0, eps, n_test=10**5, batch_size=exp.batch_size_rand)
    np.savez(rand_npz, eps=eps, bound=bound)
    t1 = time.time()
    print('time to compute:', t1-t0, 'seconds')

###############################################################################
# lower bound, gradient
if compute_grad:
    print('\nLOWER LIPSCHITZ BOUNDS, GRADIENT')
    x0.requires_grad = True
    #utils.lower_bound_FGSM(net, x0, eps_lb, lb_net_grad_npz)
    #utils.lower_bound_adv(net, x0, eps_lb, lb_net_grad_npz)
    bound = utils.lower_bound_asc(net, x0, eps, step_size=exp.step_size_grad)
    x0.requires_grad = False
    np.savez(grad_npz, eps=eps, bound=bound)


###############################################################################
# upper bound, global
if compute_global:
    print('\nUPPER LIPSCHITZ BOUNDS, GLOBAL')
    t0 = time.time()
    lip_glob = network_bound.global_bound(net, x0)
    t1 = time.time()
    bound_glob = np.prod(lip_glob)
    print('bound:', bound_glob)
    print('compute time:', t1-t0, 'seconds')
    np.savez(global_npz, bound=lip_glob)


###############################################################################
# upper bound, local
if compute_local:
    print('\nUPPER LIPSCHITZ BOUNDS, LOCAL')

    # loop over input perturbation sizes
    t0 = time.time()
    bound = [None]*n_bound
    for i, eps_i in enumerate(tqdm(eps)):
        bound[i] = network_bound.local_bound(net, x0, eps_i, batch_size=exp.batch_size_l)

    t1 = time.time()
    print('total compute time:', t1-t0, 'seconds')
    print('average compute time per epsilon', (t1-t0)/len(eps), 'seconds')

    np.savez(local_npz, eps=eps, bound=bound)
