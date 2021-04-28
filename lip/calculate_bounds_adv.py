import os
import numpy as np
import torch
from tqdm import tqdm

import my_config
import utils
import network_bound

# setup
device = my_config.device 
import mnist as exp
#import cifar10 as exp
#import alexnet as exp
#import vgg16 as exp

main_dir = exp.main_dir
net_name = exp.net_name
plot_name = exp.plot_name
net = exp.net()
net = net.to(device)
net.eval()
layers = net.layers
n_layers = len(layers)

# iterate over each layer and get the output of each function
x0 = exp.x0
x0 = x0.to(device)
layers = net.layers
n_layers = len(layers)
X = [x0]
for i in range(n_layers):
    f = layers[i]
    X.append(f(X[-1]))

# upper bound layer, global
global_npz = os.path.join(main_dir, 'global.npz')
dat = np.load(global_npz, allow_pickle=True)
bound_layer_global = dat['bound']
bound_global = np.prod(bound_layer_global)

###############################################################################
# LOCAL LIPSCHITZ LOWER BOUND
# get largest and next largest elements of output (pre-softmax)
print('\nLOCAL LOWER BOUND')

x0.requires_grad = True 
y0 = net(x0)
top2_true, ind_top2_true = torch.topk(y0.flatten(), 2)
ind1_true = ind_top2_true[0].item()
ind2_true = ind_top2_true[1].item()
output_delta = (top2_true[0]-top2_true[1]).item() # minimum change in output to change classification

# bisection method to find largest epsilon
'''

      |--------------------------|--------------------------|
   eps_min                      eps                      eps_max

'''
n_runs = 10
eps_min = exp.eps_min_bisect
eps_max = exp.eps_max_bisect
for i in range(n_runs):
    eps = (eps_max + eps_min)/2
    L_bound = network_bound.network_bound(net, x0, eps, batch_size=exp.batch_size_l)
    if eps*L_bound < output_delta/np.sqrt(2):
        eps_greatest = eps
        print('eps', eps, 'lower bound')
        print('L bound', L_bound)
        #print('new largest epsilon is ', eps_greatest)
        eps_min = eps
    else:
        print('eps', eps, 'NOT lower bound')
        print('L bound', L_bound)
        eps_max = eps
print('largest epsilon is', eps_greatest)


###############################################################################
# GLOBAL LIPSCHITZ LOWER BOUND
print('\nGLOBAL LOWER BOUND')
# L >= eps_out/eps_in
# eps_in >= eps_out/L
#eps_max_class_global = output_delta/bound_global # old, wrong
eps_max_class_global = (output_delta+np.sqrt(2))/bound_global
print('global lower epsilon bound:', eps_max_class_global)

###############################################################################
# FGSM
print('\nFGSM UPPER BOUND')

eps_list = np.arange(0,100,.01)
for eps in eps_list:
    ind_fgsm, pert_norm = utils.fgsm_new(net, x0, eps)
    if ind_fgsm != ind1_true:
        print('minimum epsilon', eps)
        print('minimum perturbation', pert_norm)
        break

if eps == eps_list[-1]:
    print('no perturbation found')

###############################################################################
# gradient ascent
# evaluate the nominal input
#x0.requires_grad = True
#y0 = net(x0)
#ind_true = top2_inds[0].item()
#ind_true_2nd = top2_inds[1].item()
#out_mask = torch.zeros(y0.shape).to(device)
#out_mask[0,ind_true_2nd] = 1

# gradient ascent
if net_name=='mnist':
    step_size = .01 # best for no FGSM
    #step_size = .0001 # best for FGSM
elif net_name=='cifar10':
    step_size = .001 # ind=2, step_size=.001
elif net_name=='alexnet':
    step_size = .01 # ind=5, step_size=.01
elif net_name=='vgg16':
    step_size = .01 # ind=3, step_size=.01

# do gradient ascent with respect to all classes to find the smallest
# perturbation that causes the classification to change
fgsm = 1

pert_min = np.inf
for ind in tqdm(range(y0.numel())):
    x_i, pert_size_i, ind_new, its_i = utils.adv_asc_class_change(net, x0, ind, step_size, fgsm=fgsm)
    if (pert_size_i is not None) and (pert_size_i<pert_min):
        pert_min = pert_size_i
        ind_min = ind_new
        print('new pert', pert_min)
        print('its', its_i)

# print
#pert_min = utils.adv_asc_class_change_batch(net, x0, step_size)
print('\nADVERSARIAL UPPER BOUND')
print('pert:', pert_min)
#print('grad:', grad)
print('old class, best:', exp.classes[ind1_true])
print('old class, 2nd best:', exp.classes[ind2_true])
print('ind:', ind_new)
print('new class:', exp.classes[ind_new])
#print('iterations:', its)

"""
###############################################################################
# random
eps = 5
eps_inc = 5
min_pert = None
while min_pert is None:
    min_pert =  utils.adv_rand_class_change(net, x0, eps=eps, n_samp=exp.batch_size_ball)
    eps = eps + eps_inc
print('\nSMALLEST RANDOM ADVERARIAL PERTURBATION FOUND', min_pert)
"""
