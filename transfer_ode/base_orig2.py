"""
base solver for transfer ode (first order methods)
"""
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torchdiffeq import odeint_adjoint as odeint
from mpl_toolkits.mplot3d import Axes3D
import random
import parser_args # import *
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import os

import sys

import f_gen
import torch
import pickle
import pandas as pd

if sys.stdin and sys.stdin.isatty():
    print("iteractive")
else:
    print("not interactive")
torch.set_default_tensor_type(torch.FloatTensor)


# parser = argparse.ArgumentParser('transfer demo')

# parser.add_argument('--tmax', type=float, default=3.)
# parser.add_argument('--dt', type=int, default=0.1)
# parser.add_argument('--niters', type=int, default=10000)
# parser.add_argument('--niters_test', type=int, default=15000)
# parser.add_argument('--hidden_size', type=int, default=100)
# parser.add_argument('--num_bundles', type=int, default=20)
# parser.add_argument('--num_bundles_test', type=int, default=1000)
# parser.add_argument('--test_freq', type=int, default=100)
# parser.add_argument('--viz', action='store_false')
# parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--evaluate_only', action='store_false')
# args = parser.parse_args()
# scaler = MinMaxScaler()

#instead of keeping the parser args localy we can externalize them
parser = parser_args.parse_args_bundles_('transfer demo')
args = parser.parse_args()
locals().update(args.__dict__)

scaler = MinMaxScaler()

def get_udot_2(t,y,yd,a1,a0,f):

    #a1 is 1
    # print(t.dim())
    if y.shape[0] <=1:
        a1s = torch.tensor([a_(t) for a_ in a1]).reshape(1, -1)
        a0s = torch.tensor([a_(t) for a_ in a0]).reshape(1,-1)
        f0s = torch.tensor([f_(t) for f_ in f]).reshape(1,-1)
    else:
        a1s = torch.cat([a_(t) for a_ in a1], 1)
        a0s = torch.cat([a_(t) for a_ in a0],1)
        f0s = torch.cat([f_(t) for f_ in f],1)

    ydd = (-a1s*yd -a0s * y + f0s)
    return ydd

class diffeq(nn.Module):
    """
    defines the diffeq of interest
    """

    def __init__(self,a1, a0, f):
        super().__init__()
        self.a1 = a1
        self.a0 = a0
        self.f = f

    # return ydot
    def forward(self, t, states):
        # y = y[:, 0]
        y = states[:, 0].reshape(1, -1)
        yd = states[:, 1].reshape(1, -1)
        ydd = get_udot(t,y,yd,self.a1,self.a0,self.f)#(-self.a1(t) * yd - self.a0(t) * y + self.f(t)).reshape(-1, 1)
        return torch.cat([yd.reshape(-1,1), ydd.reshape(-1,1)], 1)


def get_udot(t,y,yd,a1,a0,f):

    #a1 is 1
    # print(t.dim())
    if y.shape[0] <=1:
        a1s = torch.tensor([a_(t) for a_ in a1]).reshape(1, -1)
        a0s = torch.tensor([a_(t) for a_ in a0]).reshape(1,-1)
        f0s = torch.tensor([f_(t) for f_ in f]).reshape(1,-1)
    else:
        a1s = torch.cat([a_(t) for a_ in a1], 1)
        a0s = torch.cat([a_(t) for a_ in a0],1)
        f0s = torch.cat([f_(t) for f_ in f],1)

    ydd = (-a1s*yd -a0s * y + f0s)
    return ydd

class base_diffeq:
    """
    integrates base_solver given y0 and time
    """

    def __init__(self, base_solver):
        self.base = base_solver

    def get_solution(self, true_y0, t):
        with torch.no_grad():
            true_y = odeint(self.base, true_y0, t, method='dopri5')
        return true_y

    def get_deriv(self, true_y0, t):
        with torch.no_grad():
            true_ydot = self.base(t, true_y0)
        return true_ydot


class estim_diffeq:
    """
    integrates base_solver given y0 and time
    """

    def __init__(self, base_solver):
        self.base = base_solver

    def get_solution(self, true_y0, t):
        with torch.no_grad():
            true_y = odeint(self.base, true_y0, t, method='midpoint')
        return true_y

    def get_deriv(self, true_y0, t):
        with torch.no_grad():
            true_ydot = self.base(t, true_y0)
        return true_ydot


class ODEFunc(nn.Module):
    """
    function to learn the outputs u(t) and hidden states h(t) s.t. u(t) = h(t)W_out
    """

    def __init__(self, hidden_dim, output_dim, calc_bias):
        super(ODEFunc, self).__init__()
        self.hdim = hidden_dim
        self.nl = nn.Tanh()
        self.lin1 = nn.Linear(1, self.hdim)
        self.lin2 = nn.Linear(self.hdim, self.hdim)
        self.lout = nn.Linear(self.hdim, output_dim, bias = calc_bias)

    def forward(self, t):
        self.h_ = x = self.h(t)
        #self.h_.retain_grad()
        x = self.lout(x)
        return x

    def wouts(self, x):
        return self.lout(x)

    def h(self, t):
        x = self.lin1(t)
        x = self.nl(x)
        x = self.lin2(x)
        x = self.nl(x)
        return x


def diff(u, t, order=1, grad_outputs = None):
    # code adapted from neurodiffeq library
    # https://github.com/NeuroDiffGym/neurodiffeq/blob/master/neurodiffeq/neurodiffeq.py
    """The derivative of a variable with respect to another.
    """
    # ones = torch.ones_like(u)
    sum_u = u.sum(axis = 0)
    #assert sum_u.shape == grad_outputs.shape, f'{sum_u.shape} == {grad_outputs.shape}'

    der = torch.cat([torch.autograd.grad(sum_u[i], t, create_graph=True)[0] for i in range(u.shape[1])], 1)
    if der is None:
        print('derivative is None')
        return torch.zeros_like(t, requires_grad=True)
    else:
        der.requires_grad_()
    for i in range(1, order):
        sum_der = der.sum(axis = 0)
        der = torch.cat([torch.autograd.grad(sum_der[j], t, create_graph=True)[0] for j in range(der.shape[1])], 1)
        # print()
        if der is None:
            print('derivative is None')
            return torch.zeros_like(t, requires_grad=True)
        else:
            der.requires_grad_()
    return der


class Transformer_Learned(nn.Module):
    """
    returns Wout learnable, only need hidden and output dims
    """

    def __init__(self, input_dims, output_dims):
        super(Transformer_Learned, self).__init__()
        self.lin1 = nn.Linear(args.hidden_size, output_dims)

    def forward(self, x):
        return self.lin1(x)

class Transformer_Analytic(nn.Module):
    """
    returns Wout analytic, need to define the parameter coefficients
    """

    def __init__(self, lambda_, no_calc_bias):
        super(Transformer_Analytic, self).__init__()

        self.calc_bias = not no_calc_bias
        self.lambda_ = lambda_

    def calc_bias(self, weights):
        return self._y_means - self._x_means @ weights

    def _center_H(self, inputs = None, outputs = None, keep = False):
        """
        INSTRUCTIONS:
        1. assign `_x_means` to self, along the axis such that 
           the numbers of means matches the number of features (2)
        2. assign `_y_mean` to self (y.mean())
        3. subtract _x_means from X and assign it to X_centered
        4. subtract _y_mean from y and assign it to y_centered
        """
        if inputs is not None:
            X = inputs

            if keep:
                self._x_means = X.mean(axis=0)
                self._x_stds = X.std(axis = 0)

            X_centered = (X - self._x_means)#/self._x_stds
            return X_centered
        if outputs is not None:
            y = outputs

            if keep:
                self._y_means = y.mean(axis = 0)

            y_centered = y - self._y_means #(y - y_means)/y_stds
            return y_centered

    # def get_wout_ole(self, s, sd, y0, t, a0s, fs):
    #     ny0 = torch.stack([y0 for _ in range(len(s))]).reshape(len(s), -1)

    #     if self.calc_bias:
    #         ones_col = torch.ones_like(s[:,0]).view(-1,1)
    #         #states with bias
    #         s = torch.hstack((ones_col, s))

    #         #sd with zeros
    #         sd = torch.hstack((0 * ones_col, sd))

    #     na0 = torch.cat([a_(t) for a_ in a0s], 1)
    #     na1 = torch.ones_like(na0)
    #     nf = torch.cat([f_(t) for f_ in fs], 1)
    #     WS = []
    #     BS = []
    #     for i in range(nf.shape[1]):
    #         y0 = ny0[:,i].reshape(-1,1)
    #         a0 = na0[:,i].reshape(-1,1)
    #         a1 = na1[:,i].reshape(-1,1)
    #         f = nf[:,i].reshape(-1,1)
    #         D0 = -f
    #         DH = (a1*sd + a0 * s)
    #         h0m = s[0].reshape(-1, 1)
    #         LHS = DH.t() @ DH + h0m @ h0m.t()
    #         W0 = torch.linalg.solve(LHS, -DH.t() @ D0 + h0m @ (y0[0, :].reshape(1, -1)))


    #         if self.calc_bias:
    #             weight = W0[1:]
    #             bias = W0[0]
    #             W0 = weight
    #             # elif ridge:
    #             #     W0  = torch.tensor(clf.coef_, dtype = torch.float32).T
    #             #     bias = clf.intercept_
    #         else:
    #             bias = 0

    #         WS.append(W0)
    #         BS.append(bias)

    #     nWS = (torch.cat(WS)).reshape(nf.shape[1],-1).T
    #     nBS = (torch.cat(BS)).reshape(nf.shape[1],-1).T
    #     return nWS, nBS#.t()

    def get_wout(self,s = None, sd = None, sdd= None, y0s = None, t = None, a1s = None, a0s = None, fs= None):
        # y0 = torch.stack([y0 for _ in range(len(s))]).reshape(len(s), -1)

        # if self.calc_bias:
        #     ones_col = torch.ones_like(s[:,0]).view(-1,1)
        #     zeros_col = 0 * ones_col
        #     #states with bias
        #     s = torch.hstack((ones_col, s))

        #     #sd with zeros
        #     sd = torch.hstack((zeros_col, sd))
        #     sdd = torch.hstack((zeros_col, sdd))

        a0_batch = torch.cat([var_(t) for var_ in a0s], 1)
        a1_batch = torch.cat([var_(t) for var_ in a1s], 1)
        f_batch  = torch.cat([var_(t) for var_ in fs], 1)
        WS = []
        BS = []
        for i in range(f_batch.shape[1]):
            y0 = y0s[i,:].reshape(1,-1)
            a0 = a0_batch[:,i].reshape(-1,1)
            a1 = a1_batch[:,i].reshape(-1,1)
            f = f_batch[:,i].reshape(-1,1)

            # print(a0,a1,f)
            DH = (sdd + a1 * sd + a0 * s)
            D0 = -f

            h0m = s[0].reshape(-1, 1)
            h0d = sd[0].reshape(-1, 1)
            W0 = torch.linalg.solve(DH.t() @ DH + h0m @ h0m.t() + h0d @ h0d.t(),
                                    -DH.t() @ D0 + h0m @ (y0[:,0].reshape(1, -1)) + h0d @ (y0[:,1].reshape(1, -1)))

            # if self.calc_bias:
            #     weight = W0[1:]
            #     bias = W0[0]
            #     W0 = weight
            #     # elif ridge:
            #     #     W0  = torch.tensor(clf.coef_, dtype = torch.float32).T
            #     #     bias = clf.intercept_
            # else:
            #     bias = 0
            # print(W0.shape)
            WS.append(W0)
            #BS.append(bias)
        nWS = (torch.cat(WS)).reshape(f_batch.shape[1],-1)
        #nBS = (torch.cat(BS)).reshape(f_batch.shape[1],-1)
        return nWS.t()#, nBS.t()

        

        # a0 = a0s(t).reshape(-1, 1)
        # a1 = torch.ones_like(a0)
        # f = torch.cat([f_(t) for f_ in fs], 1)

        # # a0 = torch.cat([a_(t) for a_ in a0s], 1)
        # # f0 = torch.cat([f_(t) for f_ in fs], 1)
        # # a1 = torch.ones_like(a0)
        # # idms = torch.ones((s.shape[1],a0.shape[1]))
        # D0 = -f

        # DH = (a1*sd + a0 * s)

        # # if ridge:
        # #     global DH_means
        # #     DH_means = DH.mean(axis = 0)
        # #     DH = DH- DH_means
        # h0m = s[0].reshape(-1, 1)


        # LHS = DH.t() @ DH + h0m @ h0m.t()
        # RHS = -DH.t() @ D0 + h0m @ (y0[0, :].reshape(1, -1))

        # #alphas=[1e-3, 1e-2, 1e-1, 1]

        # #breakpoint()
        

        # if self.lambda_:
        #     LHS = LHS + self.lambda_ * torch.eye(LHS.shape[0])
        # if ridge:
        #     from sklearn.linear_model import RidgeCV

        #     clf = RidgeCV(fit_intercept = True).fit(LHS, RHS) 
            

        # else:
        #self.lambda_ = 100

        # W0 = torch.linalg.solve(LHS, RHS)

        # #W0 = torch.linalg.solve( LHS , -DH.T @ D0 + h0m @ (y0[0, :].reshape(1, -1)))

        
        # return W0, bias




# def get_wout(s, sd, y0, t,a0s,fs):

#     y0 = torch.stack([y0 for _ in range(len(s))]).reshape(len(s), -1)

#     a0 = a0s(t).reshape(-1, 1)
#     a1 = torch.ones_like(a0)
#     f = torch.cat([f_(t) for f_ in fs], 1)

#     # a0 = torch.cat([a_(t) for a_ in a0s], 1)
#     # f0 = torch.cat([f_(t) for f_ in fs], 1)
#     # a1 = torch.ones_like(a0)
#     # idms = torch.ones((s.shape[1],a0.shape[1]))
#     D0 = -f

#     DH = (a1*sd + a0 * s)
#     h0m = s[0].reshape(-1, 1)

#     W0 = torch.linalg.solve(DH.t() @ DH + h0m @ h0m.t(), -DH.t() @ D0 + h0m @ (y0[0, :].reshape(1, -1)))
#     return W0



#     # y0 = torch.stack([y0 for _ in range(len(s))]).reshape(len(s), -1)

#     # a0 = a0s(t).reshape(-1, 1)
#     # a1 = torch.ones_like(a0)
#     # f = torch.cat([f_(t) for f_ in fs], 1)

#     # # a0 = torch.cat([a_(t) for a_ in a0s], 1)
#     # # f0 = torch.cat([f_(t) for f_ in fs], 1)
#     # # a1 = torch.ones_like(a0)
#     # # idms = torch.ones((s.shape[1],a0.shape[1]))
#     # D0 = -f

#     # DH = (a1*sd + a0 * s)
#     # h0m = s[0].reshape(-1, 1)

#     # W0 = torch.linalg.solve(DH.t() @ DH + h0m @ h0m.t(), -DH.t() @ D0 + h0m @ (y0[0, :].reshape(1, -1)))
#     # bias = self.calc_bias(W0)
#     # breakpoint()
#     # return W0, bias





#     # right_term = torch.einsum('ik,il->ilk', a0, s)
#     # left_term = torch.einsum('ik,il->ilk', torch.ones_like(a0), sd)
#     # DH = (left_term + right_term)
#     # D0 = -f0
#     #
#     # DH = torch.einsum('ilk->kil',DH)
#     # DHt = torch.einsum('kil->kli',DH)
#     #
#     # DHtDH = torch.einsum('kli,kil->kll',DHt,DH)
#     # h0m = s[0].reshape(-1, 1)
#     # W0 = torch.linalg.solve(DHtDH+ h0m @ h0m.t(), -DH.t() @ D0 + h0m @ (y0[0, :].reshape(1, -1)))
#     # return W0

#     # a0 = a0s(t).reshape(-1, 1)
#     # a1 =1.
#     # # f = fs(t).reshape(-1, 1)
#     # f = torch.cat([f_(t) for f_ in fs], 1)
#     #
#     # DH = (a1 * sd + a0 * s)
#     # D0 = (-f).repeat_interleave(y0.shape[1]).reshape(-1, y0.shape[1])
#     # lambda_0 = self.lambda_
#     #
#     # h0m = s[0].reshape(-1, 1)
#     # W0 = torch.linalg.solve(DH.t() @ DH + lambda_0 + h0m @ h0m.t(), -DH.t() @ D0 + h0m @ (y0[0, :].reshape(1, -1)))
#     # return W0
import matplotlib.pyplot as plt

if args.viz:
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)

def visualize(tv, true_y, pred_y, lst = None):
    

    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        # for i in range(args.num_bundles):
        #     ax_traj.plot(tv.detach().cpu().numpy(), true_y.cpu().numpy()[:, i],
        #                  'g-')
        #     ax_traj.plot(tv.detach().cpu().numpy(), pred_y.cpu().numpy()[:, i], '--', 'b--')
        plt.plot(lst)
        plt.yscale('log')
        ax_phase.set_yscale('log')
        #ax_phase.plot(np.arange(len(lst)), lst)

        ax_traj.legend()

        plt.draw()
        plt.pause(0.001)
        #plt.show()

# def optimize(a0 = lambda t: t**2,#-(5./t + t)#-3*t**2
#              a1 = lambda t:1 + 0.*t,
#              f = lambda t: torch.sin(t),#t**6#3*t**2#torch.sin(t)
#              ics = torch.tensor(np.arange(-2.9, 2.9, 0.1), dtype = torch.float32),#torch.linspace(-7.,7.,200),
#              method : str = "dopri5", 
#              tmax : float = 5,
#              #dt   : int   = 0.01,
#              method_rc: str = "euler",
#              wout : str = "analytic",
#              paramg : str = "lin",
#              niters : int = 100,
#              hidden_size : int = 200,
#              viz = False,#'store_false',
#              gpu : int = 0,
#              adjoint = 'store_false',
#              random_sampling = True,
#              n_timepoints = 50,
#              regularization = 0,
#              l1_reg_strength = 0,
#              #visualize_ = False,
#              niters_test: int =15000,
#              num_bundles: int= 20,
#              num_bundles_test : int =20,
#              test_freq :int =10,
#              evaluate_only : bool = False,
#              bias_at_inference : bool = False,
#              ffnn_bias: bool = False,
#              force_bias : int  = 0
#             ):

#if args.viz:

def get_true_y(a1s, a0s, fs, y0s, t):
    diffeq_init = diffeq(a1s, a0s, fs)
    gt_generator = base_diffeq(diffeq_init)
    true_y = gt_generator.get_solution(y0s, t.ravel())
    true_y = true_y[:,:,0]
    return true_y
def optimize(ic_train_range, ic_test_range, filename):

    ic_train_range = [int(num) for num in ic_train_range]
    ic_test_range = [int(num) for num in ic_test_range]
    # args = Args()
    
    # args.assign(locals())
    # args = args
    #assert False, args.__dict__
    # print(f' bias_at_inference {not no_bias_at_inference}, ridge {ridge}')
    # print(f' bias_at_inference {ffnn_bias}, ridge {ridge}')
    if args.wout == 'analytic':
        wout_gen = Transformer_Analytic(regularization, no_bias_at_inference)
        #wout_gen = Transformer_Analytic(a0, a1, f, regularization)
        
    tmax = np.pi
    
    dt=tmax/n_timepoints
    args.dt = dt
    
    #if not random_sampling:
    t = torch.arange(0.,tmax,args.dt)
    assert len(t) == args.n_timepoints, f'{len(t)} != {args.n_timepoints}'
    # else:
    #     t = torch.rand(n_timepoints) *tmax
    #     t = t.sort().values
    
    t = t.reshape(-1,1)
    
    #assign_vars(compute_s_sdot, "t", t)
    
    
    globals()["args"] = args
    
    ii = 0
    NDIMZ = args.hidden_size
    # define coefficients as lambda functions, used for gt and wout_analytic
    # training differential equation

    #need to sample tuple of (a1,f,IC)
    # each column of Wouts defines a solution thus, each tuple defines a solution too

    hp1 = phase_shift = 2*np.pi
    hp2 = amplitude_range = 5
    hp3 = angular_freq_range = 3

    num_forces = max(num_bundles//5, 1)

    # f_generator = f_gen.Wave_Gen(phase_shift = hp1, amplitude_range = hp2, angular_freq_range =hp3)

    # f_train = [torch.sin, torch.cos] +[ f_generator.realize_recursive() for _ in range(num_forces)] #+ [ f_generator.realize() for _ in range(n_forces)]
    # f_train = [lambda t: torch.cos(t) + force_bias,
    #            lambda t: torch.cos(t) - force_bias,
    #            lambda t: torch.sin(t) - force_bias, 
    #            lambda t: torch.sin(t) + force_bias, 
    #            lambda t: torch.sin(t)* torch.cos(t) - force_bias,
    #            lambda t: torch.sin(t)* torch.cos(t) + force_bias]

    a0_train = [lambda t: torch.zeros_like(t), lambda t: torch.ones_like(t), lambda t: t, lambda t: t**2, lambda t: t**3]#, lambda t : torch.exp(-t)] #[lambda t:t**2]
    a1_train = [lambda t: torch.zeros_like(t), lambda t: torch.ones_like(t), lambda t: t, lambda t: t**2, lambda t: t**3]
    r1 = ic_train_range[0]
    r2 = ic_train_range[1]

    true_y0 = (r2 - r1) * torch.rand(100, 2) + r1
    t = torch.arange(0., args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True

    # sample each parameter to build the tuples
    f_samples = random.choices(f_train, k=args.num_bundles)
    a0_samples = random.choices(a0_train, k=args.num_bundles)
    a1_samples = random.choices(a1_train, k=args.num_bundles)
    #y0_samples = torch.tensor(random.choices(true_y0, k=args.num_bundles)).reshape(1,-1)
    idx = torch.tensor(random.choices(range(len(true_y0)), k=args.num_bundles))
    y0_samples = true_y0[idx]

    diffeq_init = diffeq(a1_samples,a0_samples,f_samples)
    gt_generator = base_diffeq(diffeq_init)
    true_y = gt_generator.get_solution(y0_samples,t.ravel())
    
    # use this quick test to find gt solutions and check training ICs
    # have a solution (don't blow up for dopri5 integrator)
    # true_y = gt_generator.get_solution(true_y0.reshape(-1, 1), t.ravel())

    # instantiate wout with coefficients
    func = ODEFunc(hidden_dim=NDIMZ, output_dim=args.num_bundles, calc_bias = ffnn_bias)

    lr = 1e-3

    optimizer = optim.Adam(func.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.5)

    optimizer2 = optim.SGD(func.parameters(), lr = 1e-5, weight_decay = 1e-6, momentum = 0.8)

    loss_collector = []
    # if not "exp_name" in locals().keys():
    #     exp_name = ""

    # bp = "../results/"

    # try:
    #     os.mkdir(bp)
    # except:
    #     pass
    # bp += 'func_ffnn_bundles/'
    # try:
    #     os.mkdir(bp)
    # except:
    #     pass
    # experiment_name = "t2/"
    # bp +=experiment_name
    # try:
    #     os.mkdir(bp)
    # except:
    #     pass

    # filename = bp + "__num_bundles_"+  str(num_bundles) + "__num_forces_" + str(num_forces)
    # exp_name = filename  + ".pt"


    

    a0_test = a0_train + coefs_test
    a1_test = a1_train + coefs_test
    #if args.viz:
    spikethreshold = 0.2


    if not args.evaluate_only:
        best_residual = np.inf
        lrs = []

        for itr in range(1, args.niters + 1):
            func.train()



            # add t0 to training times, including randomly generated ts
            t0 = torch.tensor([[0.]])
            t0.requires_grad = True
            tv = args.tmax * torch.rand(int(args.tmax / args.dt)).reshape(-1, 1)
            tv.requires_grad = True
            tv = torch.cat([t0, tv], 0)
            optimizer.zero_grad()

            # compute hwout,hdotwout
            # h = func.h(tv)
            # hd = diff(h, tv)

            #h = func.h(tv)
            


            # h = func.h(tv)# @ .weight.T
            # pred_y = func.lout(h)

            pred_y = func(tv)
            pred_ydot = diff(pred_y, tv)
            pred_yddot = diff(pred_ydot,tv)

            #hd = diff(h, tv, grad_outputs = func.lout.weight.T)

            #pred_ydot = hd @ func.lout.weight.T
            #pred_ydot.retain_grad()
            
            udot = get_udot(tv,pred_y,pred_ydot,a1_samples,a0_samples,f_samples)
            loss_diffeq = pred_yddot - udot

            # enforce diffeq
            loss_ics = torch.mean((pred_y[0, :].ravel() - y0_samples[:,0].ravel())**2) + torch.mean((pred_ydot[0,:].ravel()-y0_samples[:,1].ravel())**2)
            L1 ,L2 = torch.mean(torch.square(loss_diffeq)), torch.mean(loss_ics)
            loss = L1 + L2


            loss.backward()
            optimizer.step()

            L3 = L1.mean().item()

            loss_collector.append(L3)

            if itr % args.test_freq == 0:
                func.eval()

                pred_y = func(t)
                pred_ydot = diff(pred_y,t)
                pred_yddot = diff(pred_ydot,t)

                pred_y = pred_y.detach()
                pred_ydot = pred_ydot.detach()
                pred_yddot = pred_yddot.detach()
                current_residual = torch.mean((pred_yddot - get_udot(t,pred_y,pred_ydot,a1_samples,a0_samples,f_samples))**2)

                pred_y_ = pred_y[:,1:].detach()
                pred_ydot = pred_ydot[:,1:].detach()


                
                #print(current_residual.item())
                if current_residual < best_residual:
                    best_weight_state_dict = func.state_dict()
                    #torch.save(func.state_dict(), 'func_ffnn_bundles')
                    best_residual = current_residual
                    print(itr,best_residual.item())

                
    y0_val_samples = true_y0[torch.tensor(random.choices(range(len(true_y0)), k=args.num_bundles_test))]

    #assert False, y0_val_samples.shape
    f_val_samples = random.choices(f_train, k=args.num_bundles_test)
    a0_val_samples = random.choices(a0_train, k=args.num_bundles_test)
    a1_val_samples = random.choices(a1_train, k=args.num_bundles_test)
    val_samples = {"fs" : f_val_samples, 
                    "a0s": a0_val_samples, 
                    "a1s": a1_val_samples, 
                    "y0s": y0_val_samples}

    r1 = ic_test_range[0]
    r2 = ic_test_range[1]
    y0_test = (r2 - r1) * torch.rand(100,2) + r1
    args.tmax = np.pi
    t = torch.arange(0., args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True

    # sample each parameter to build the tuples
    y0_test_samples = true_y0[torch.tensor(random.choices(range(len(y0_test)), k=args.num_bundles_test))]
    f_test_samples = random.choices(f_test, k=args.num_bundles_test)
    a0_test_samples = random.choices(a0_test, k=args.num_bundles_test)
    a1_test_samples = random.choices(a1_test, k=args.num_bundles_test)
    test_samples = {"fs" : f_test_samples, 
                    "a0s": a0_test_samples, 
                    "a1s": a1_test_samples, 
                    "y0s": y0_test_samples}    

    if not args.save:
        func.load_state_dict(torch.load(filename ))


    func.eval()

    h = func.h(t)
    hd = diff(h, t)
    hdd = diff(hd, t)

    h = h.detach()
    hd = hd.detach()
    hdd = hdd.detach()

    h = torch.cat([h,torch.ones(len(h),1)],1)
    hd = torch.cat([hd,torch.zeros(len(hd),1)],1)
    hdd = torch.cat([hdd,torch.zeros(len(hdd),1)],1)


    common_args = { "t" : t, "s" : h, "sd" : hd, "sdd" : hdd}

    s1 = time.time()

    ############VALIDATION

    val_args = {**val_samples, **common_args}
    wout = wout_gen.get_wout(**val_args)#h, hd, hdd, true_y0, t.detach(), a1_samples, a0_samples, f_samples)#h, hd, y0_samples, t.detach(), a0_samples, f_samples)
    
    pred_y_val = h@wout
    pred_yd = hd@wout
    pred_ydd = hdd @ wout
    
    final_val_residual = torch.mean((pred_ydd - get_udot(t, pred_y_val, pred_yd, a1_val_samples, a0_val_samples, f_val_samples)) ** 2)

    print(f'final val residual' , final_val_residual)
    ############TEST


    wout  = wout_gen.get_wout(**test_samples, **common_args)

    pred_y_test = h@wout
    pred_yd = hd@wout
    pred_ydd = hdd @ wout
    
    final_test_residual = torch.mean((pred_ydd - get_udot(t, pred_y_test, pred_yd, a1_test_samples, a0_test_samples, f_test_samples)) ** 2)

    print(f'final test residual' , final_test_residual)

    s2 = time.time()
    print(f'all_ics:{s2 - s1}')

    s1 = time.time()
    
    true_y_val  = get_true_y(**val_samples, t = t)
    true_y_test = get_true_y(**test_samples, t = t)
    
    val_resids = (pred_y_val - true_y_val) ** 2
    test_resids = (pred_y_test - true_y_test) ** 2

    val_score  = val_resids.mean()
    test_score  = test_resids.mean()

    print(current_residual)
    
    s2 = time.time()
    print(f'gt_ics:{s2 - s1}')

    gz_np = h.detach().numpy()
    T = np.linspace(0, 1, len(gz_np)) ** 2
    new_hiddens = scaler.fit_transform(gz_np)

    if plot_pca or plot_tsne:

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        from sklearn.manifold import TSNE
        if plot_tsne:
            pca = PCA(n_components=plot_tsne)
        else:
            pca = PCA(n_components=3)

        comps = pca.fit_transform(new_hiddens)

        if plot_tsne:
            comps = TSNE(n_components=3).fit_transform(comps)
            comps = comps[comps[:, 0].argsort()]

        
        if comps.shape[1] >= 2:
            s = 10  # Segment length
            for i in range(0, len(gz_np) - s, s):

                if plot_tsne:
                    ax.plot3D(comps[i:i + s + 1, 0], comps[i:i + s + 1, 1], comps[i:i + s + 1, 2],
                              color=(0.1, 0.8, T[i]))
                else:

                    ax.plot3D(comps[i:i + s + 1, 0], comps[i:i + s + 1, 1], comps[i:i + s + 1, 2],
                              color=(0.1, 0.8, T[i]))
                plt.xlabel('comp1')
                plt.ylabel('comp2')


    # s1 = time.time()
    # true_y = estim_generator.get_solution(ics.reshape(-1, 1), t.ravel())
    # estim_ys = true_y.reshape(len(pred_y), ics.shape[1])
    # s2 = time.time()
    # print(f'estim_ics:{s2 - s1}')

    print(f'val_accuracy:{val_score} pm {val_resids.std()}')
    print(f'test_accuracy:{test_score} pm {test_resids.std()}')

    #rint(f'estim_accuracy:{((estim_ys - true_ys) ** 2).mean()} pm {((estim_ys - true_ys) ** 2).std()}')

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # print(true_ys[0,:])
    for i in range(0, args.num_bundles_test, 50):
        gt = true_y_val.cpu().numpy()[:, i]
        preds = pred_y_val.cpu().detach().numpy()[:, i]
        ax[0].plot(t.detach().cpu().numpy(), gt, c='blue', linestyle='dashed')
        ax[0].plot(t.detach().cpu().numpy(),  preds , c='orange')
        # plt.draw()

    ax[1].plot(t.detach().cpu().numpy(), ((true_y_val - pred_y_val) ** 2).mean(1).cpu().detach().numpy(), c='green')
    ax[1].set_xlabel('Time (s)')
    #plt.legend()
    #plt.show()

    if args.save:

        fig.savefig(filename+str(num_bundles)+  "_pca")
    
    
    #estimation_residuals = ((estim_ys - true_ys) ** 2)
    
    return best_weight_state_dict, val_score, test_score, pred_y_val, pred_y_test, true_y_val, true_y_test, filename, loss_collector, h, best_residual


if __name__ == "__main__":



    bp = '../results/func_ffnn_bundles/t2/'
    #spec_fp = "final/500_timepoints/"
    spec_fp ="second_order/"
    filename = bp+spec_fp+"bundles_final_500_timepoints"#"bundles_sd_final"



    #model, score, pred_y, true_y, filename, loss_collector = optimize(ic_tr_range, ic_te_range)

    # results["models"].append(model)
    # results["scores"].append(score)
    # results["pred_ys"].append(pred_y)
    # results["true_ys"].append(true_y)

    # with open(filename + '.pickle', 'wb') as f:
    #     pickle.dump(results, f)

    #evaluate_only = False
    #score, pred_y, true_y, filename = optimize(ic_tr_range, ic_te_range)
    results = {"models" : [], "val_scores" : [], "test_scores" : [],
                "pred_y_val" : [], "pred_y_test" : [], "gt_val_y" : [], "gt_test_y" : [],
                "loss" : [], "hs" : [], 'tr_scores' : [], 'num_bundles' : [],
                "n_timepoints" : [], "n_iters" : []}
    timepoints = args.n_timepoints
    n_iters = args.niters
    for _ in range(1):
        hp1 = phase_shift = 1*np.pi
        hp2 = amplitude_range = 2
        hp3 = angular_freq_range = 2
        f_generator = f_gen.Wave_Gen(phase_shift = hp1, amplitude_range = hp2, angular_freq_range =hp3)

        f_train = [lambda t: torch.ones_like(t), torch.sin, torch.cos, lambda t: torch.sin(2*t), lambda t: torch.cos(2*t), lambda t: torch.sin(t) * torch.cos(t)]#
            
        coefs_test = [ lambda t : t**1.5 + 1,
                       lambda t :( t + 0.2)**0.5,
                       lambda t : t**2.5,
                       lambda t : 2* t**2 - 2*t + 1,
                       lambda t: t**2 * torch.exp(-t) ]

        #wave_gen_test = f_gen.Wave_Gen(phase_shift = hp1, amplitude_range = hp2, angular_freq_range =hp3)

        f_test = f_train + [ lambda t: 0.1 * torch.sin(0.2 * t), 
                             lambda t: -0.1 * torch.sin(0.2 * t), 
                             
                             lambda t: torch.sin(3*t),
                             lambda t: torch.cos(3*t)]+ [ f_generator.realize_recursive() for _ in range(5)] 
        for n in [1, 5, 10, 20, 50, 100, 500]:# 1000]:range(1,10):
            assert ffnn_bias, 'asdf'
            #for ratio in [1, 2, 10]:#[1, 2, 10]:
            ratio = 1
            num_forces = max(n//ratio,1)
            args.num_bundles = num_bundles = n

            print(f'NUM BUNDLES: {num_bundles}')
            rez = optimize(ic_tr_range, ic_te_range, filename)
            model, val_score, test_score, pred_val_y, pred_y_test, true_y_val, true_y_test, filename, loss_collector, h, b_resid = rez
            results["models"].append(model)
            results["val_scores"].append(val_score)
            results["test_scores"].append(test_score)
            results["num_bundles"].append(num_bundles)
            results["pred_y_val"].append(pred_val_y)
            results["pred_y_test"].append(pred_y_test)
            results["gt_val_y"].append(true_y_val)
            results["gt_test_y"].append(true_y_test)
            results["loss"].append(loss_collector)
            results["hs"].append(h)
            results["tr_scores"].append(b_resid)
            results["n_timepoints"].append(n_timepoints)
            results["n_iters"].append(n_iters)
    
    #filename = "500_timepoints"
    with open(filename+'.pickle', 'wb') as f:
        pickle.dump(results, f)
    