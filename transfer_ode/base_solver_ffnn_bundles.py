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

from types import SimpleNamespace



if sys.stdin and sys.stdin.isatty():
    print("iteractive")
else:
    print("not interactive")



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
# parser = parser_args.parse_args_bundles_('transfer demo')
# args = parser.parse_args()
# locals().update(args.__dict__)

# scaler = MinMaxScaler()


class diffeq(nn.Module):
    """
    defines the diffeq of interest
    """

    def __init__(self, a0, f):
        super().__init__()
        # self.a1 = a1
        self.a0 = a0
        self.f = f

    # return ydot
    def forward(self, t, y):
        # y = y[:, 0]
        yd = get_udot(t,y,self.a0,self.f)#(-self.a0(t) * y + self.f(t)) / self.a1(t)
        return yd


def get_udot(t,y,a,f):

    #a1 is 1
    # print(t.dim())
    if y.shape[0] <=1:
        a0 = torch.tensor([a_(t) for a_ in a]).reshape(1,-1)
        f0 = torch.tensor([f_(t) for f_ in f]).reshape(1,-1)
    else:
        a0 = torch.cat([a_(t) for a_ in a],1)
        f0 = torch.cat([f_(t) for f_ in f],1)

    yd = (-a0 * y + f0)
    return yd


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

    def get_wout(self, s, sd, y0, t, a0s, fs):
        ny0 = torch.stack([y0 for _ in range(len(s))]).reshape(len(s), -1)

        if self.calc_bias:
            ones_col = torch.ones_like(s[:,0]).view(-1,1)
            #states with bias
            s = torch.hstack((ones_col, s))

            #sd with zeros
            sd = torch.hstack((0 * ones_col, sd))

        na0 = torch.cat([a_(t) for a_ in a0s], 1)
        na1 = torch.ones_like(na0)
        nf = torch.cat([f_(t) for f_ in fs], 1)
        WS = []
        BS = []
        for i in range(nf.shape[1]):
            y0 = ny0[:,i].reshape(-1,1)
            a0 = na0[:,i].reshape(-1,1)
            a1 = na1[:,i].reshape(-1,1)
            f = nf[:,i].reshape(-1,1)
            D0 = -f
            DH = (a1*sd + a0 * s)
            h0m = s[0].reshape(-1, 1)
            LHS = DH.t() @ DH + h0m @ h0m.t()
            try:
                W0 = torch.linalg.solve(LHS, -DH.t() @ D0 + h0m @ (y0[0, :].reshape(1, -1)))
            except:
                W0 = torch.linalg.solve(LHS + 1e-3*torch.eye(LHS.shape[0]), -DH.t() @ D0 + h0m @ (y0[0, :].reshape(1, -1)))


            if self.calc_bias:
                weight = W0[1:]
                bias = W0[0]
                W0 = weight
                # elif ridge:
                #     W0  = torch.tensor(clf.coef_, dtype = torch.float32).T
                #     bias = clf.intercept_
            else:
                bias = 0

            WS.append(W0)
            BS.append(bias)

        nWS = (torch.cat(WS)).reshape(nf.shape[1],-1).T
        nBS = (torch.cat(BS)).reshape(nf.shape[1],-1).T
        return nWS, nBS#.t()
        

        a0 = a0s(t).reshape(-1, 1)
        a1 = torch.ones_like(a0)
        f = torch.cat([f_(t) for f_ in fs], 1)

        # a0 = torch.cat([a_(t) for a_ in a0s], 1)
        # f0 = torch.cat([f_(t) for f_ in fs], 1)
        # a1 = torch.ones_like(a0)
        # idms = torch.ones((s.shape[1],a0.shape[1]))
        D0 = -f

        DH = (a1*sd + a0 * s)

        # if ridge:
        #     global DH_means
        #     DH_means = DH.mean(axis = 0)
        #     DH = DH- DH_means
        h0m = s[0].reshape(-1, 1)


        LHS = DH.t() @ DH + h0m @ h0m.t()
        RHS = -DH.t() @ D0 + h0m @ (y0[0, :].reshape(1, -1))

        #alphas=[1e-3, 1e-2, 1e-1, 1]

        #breakpoint()
        

        # if self.lambda_:
        #     LHS = LHS + self.lambda_ * torch.eye(LHS.shape[0])
        # if ridge:
        #     from sklearn.linear_model import RidgeCV

        #     clf = RidgeCV(fit_intercept = True).fit(LHS, RHS) 
            

        # else:
        #self.lambda_ = 100

        W0 = torch.linalg.solve(LHS, RHS)

        #W0 = torch.linalg.solve( LHS , -DH.T @ D0 + h0m @ (y0[0, :].reshape(1, -1)))

        
        return W0, bias

class BayesOpt:
    pass

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

# if args.viz:
#     fig = plt.figure(figsize=(12, 4), facecolor='white')
#     ax_traj = fig.add_subplot(131, frameon=False)
#     ax_phase = fig.add_subplot(132, frameon=False)
#     ax_vecfield = fig.add_subplot(133, frameon=False)
#     plt.show(block=False)

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

hp1 = phase_shift = 1*np.pi
hp2 = amplitude_range = 2
hp3 = angular_freq_range = 2
f_generator = f_gen.Wave_Gen(phase_shift = hp1, amplitude_range = hp2, angular_freq_range =hp3)

f_train = [torch.sin, torch.cos, lambda t: t, lambda t: torch.ones_like(t), lambda t: 1+ t, lambda t: torch.exp(-t)* torch.sin(t)] +[ f_generator.realize_recursive() for _ in range(3)] 
    

def optimize(ic_train_range, ic_test_range, hidden_size, spikethreshold, lr = 1e-3, args = None, gamma = 0.5):

    #assert False, f'{kwargs} kwargs'
    # for key, val in kwargs.items():
    #     if key != 'self':
    #         locals()[key] = val

    #globals()["args"] = args
    scaler = MinMaxScaler()
    args_ = SimpleNamespace(**args)
    args = args_

    ic_train_range = [int(num) for num in ic_train_range]
    ic_test_range = [int(num) for num in ic_test_range]
    
    if args_.wout == 'analytic':
        wout_gen = Transformer_Analytic(args.regularization, args.no_bias_at_inference)
        #wout_gen = Transformer_Analytic(a0, a1, f, regularization)
    
    dt=args.tmax/args.n_timepoints

    hidden_size = int(hidden_size)
    
    if not args.random_sampling:
        t = torch.arange(0.,args.tmax,args.dt)
    else:
        t = torch.rand(args.n_timepoints) * args.tmax
        t = t.sort().values
    
    t = t.reshape(-1,1)
    
    #assign_vars(compute_s_sdot, "t", t)
    
    
    
    
    ii = 0
    NDIMZ = hidden_size
    # define coefficients as lambda functions, used for gt and wout_analytic
    # training differential equation

    #need to sample tuple of (a1,f,IC)
    # each column of Wouts defines a solution thus, each tuple defines a solution too

    hp1 = phase_shift = 2*np.pi
    hp2 = amplitude_range = 5
    hp3 = angular_freq_range = 3

    num_forces = max(args.num_bundles//5, 1)

    # f_generator = f_gen.Wave_Gen(phase_shift = hp1, amplitude_range = hp2, angular_freq_range =hp3)

    # f_train = [torch.sin, torch.cos] +[ f_generator.realize_recursive() for _ in range(num_forces)] #+ [ f_generator.realize() for _ in range(n_forces)]
    # f_train = [lambda t: torch.cos(t) + force_bias,
    #            lambda t: torch.cos(t) - force_bias,
    #            lambda t: torch.sin(t) - force_bias, 
    #            lambda t: torch.sin(t) + force_bias, 
    #            lambda t: torch.sin(t)* torch.cos(t) - force_bias,
    #            lambda t: torch.sin(t)* torch.cos(t) + force_bias]

    a0_train = [lambda t: torch.ones_like(t), lambda t: t + 1, lambda t: t, lambda t: t**2, lambda t: t**3] #[lambda t:t**2]

    r1 = ic_train_range[0]
    r2 = ic_train_range[1]

    true_y0 = (r2 - r1) * torch.rand(100) + r1
    t = torch.arange(0., args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True

    # sample each parameter to build the tuples
    f_samples = random.choices(f_train, k=args.num_bundles)
    a0_samples = random.choices(a0_train, k=args.num_bundles)
    y0_samples = torch.tensor(random.choices(true_y0, k=args.num_bundles)).reshape(1,-1)

    diffeq_init = diffeq(a0_samples,f_samples)
    gt_generator = base_diffeq(diffeq_init)
    true_y = gt_generator.get_solution(y0_samples,t.ravel()).reshape(-1,args.num_bundles)

    # use this quick test to find gt solutions and check training ICs
    # have a solution (don't blow up for dopri5 integrator)
    # true_y = gt_generator.get_solution(true_y0.reshape(-1, 1), t.ravel())

    # instantiate wout with coefficients
    func = ODEFunc(hidden_dim=NDIMZ, output_dim=args.num_bundles, calc_bias = args.ffnn_bias)

    optimizer = optim.Adam(func.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = gamma)

    loss_collector = []
    if not "exp_name" in locals().keys():
        exp_name = ""

    bp = "../results/"

    try:
        os.mkdir(bp)
    except:
        pass
    bp += 'func_ffnn_bundles/'
    try:
        os.mkdir(bp)
    except:
        pass
    experiment_name = "t2/"
    bp +=experiment_name
    try:
        os.mkdir(bp)
    except:
        pass

    filename = bp + "__num_bundles_"+  str(args.num_bundles) + "__num_forces_" + str(num_forces)
    exp_name = filename  + ".pt"

    #if args.viz:
    


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
            #pred_ydot = diff(pred_y,tv)
            pred_ydot = diff(pred_y, tv, grad_outputs = func.lout.weight.T)
            
            second_order = False

            if not second_order:
                
                udot = get_udot(tv,pred_y,a0_samples,f_samples)
                loss_diffeq = pred_ydot - udot 
            else:
            
                pred_yddot = diff(pred_ydot,tv)

                #enforce diffeq
                loss_diffeq = pred_yddot + 1 * pred_ydot + (a0(tv.detach()).reshape(-1, 1)) * pred_y - f(tv.detach()).reshape(-1, 1)



            

            # pred_ydot = diff(pred_y, tv, grad_outputs = func.lout.weight.T)

            # pred_ydot = diff(pred_y,tv)
            # pred_yddot = diff(pred_ydot,tv)

            # #enforce diffeq
            # loss_diffeq = pred_yddot + (a1(tv.detach()).reshape(-1, 1)) * pred_ydot + (a0(tv.detach()).reshape(-1, 1)) * pred_y - f(tv.detach()).reshape(-1, 1)


            # #hd = diff(h, tv, grad_outputs = func.lout.weight.T)

            # #pred_ydot = hd @ func.lout.weight.T
            # #pred_ydot.retain_grad()
            
            # udot = get_udot(tv,pred_y,a0_samples,f_samples)

            # # enforce diffeq
            # loss_diffeq = pred_ydot - udot #get_udot(tv,pred_y,a0_samples,f_samples)
            # loss_diffeq = (a1(tv.detach()).reshape(-1, 1)) * pred_ydot + (a0(tv.detach()).reshape(-1, 1)) * pred_y - f(
            #     tv.detach()).reshape(-1, 1)

            # enforce initial conditions
            loss_ics = pred_y[0, :].ravel() - y0_samples.ravel()

            L1 = torch.mean(torch.square(loss_diffeq)) 
            L2 = torch.mean(torch.square(loss_ics))

            

            loss = L1 + L2 
            loss.backward()
            optimizer.step()
            L3 = torch.square(loss_diffeq).mean().item()

            loss_collector.append(L3)

            if itr % args.test_freq == 0:
                func.eval()

                # pred_y_ = pred_y[:,1:].detach()
                #assert False, pred_y.shape
                # pred_ydot = pred_ydot[:,1:].detach()
                assert pred_y[1:,:].shape == true_y.shape, f'{pred_y.shape} != {true_y.shape}'

                # pred_y = pred_y.reshape(-1, args.num_bundles)
                #visualize(t.ravel(), true_y.detach(), pred_y[1:,:].detach(), loss_collector)
                ii += 1

                current_residual = torch.mean((pred_ydot - udot)**2)
                #print(current_residual.item())
                if current_residual < best_residual:
                    #print("saving")

                    torch.save(func.state_dict(), 'func_ffnn_bundles')
                    best_residual = current_residual
                    print(itr,best_residual.item())
                elif itr > 1: 
                    if np.log(float(prev_step_loss))  - np.log(float(L1)) > spikethreshold:
                        lrs.append(optimizer.param_groups[0]["lr"])
                        scheduler.step()

            prev_step_loss = float(L1)
            # if itr % args.test_freq == 0:
            #     func.eval()
            #     pred_y = func(t).detach()
            #     pred_y = pred_y.reshape(-1, args.num_bundles)
            #     # if args.viz:
            #     #     visualize(t, true_y.detach(), pred_y.detach(), loss_collector)
            #     ii += 1
        #print("saving", func.state_dict() )
        torch.save(func.state_dict(), exp_name)

    # with torch.no_grad():
    scale_factor = 1.2
    hp1, hp2, hp3 = [hp *scale_factor for hp in [hp1, hp2, hp3]]

    #wave_gen_test = f_gen.Wave_Gen(phase_shift = hp1, amplitude_range = hp2, angular_freq_range =hp3)

    f_test = f_train#[ wave_gen_test.realize_recursive() for _ in range(num_bundles_test)] #[lambda t: torch.sin(t)*torch.cos(t)]+
    a0_test = a0_train #
    r1 = ic_test_range[0]
    r2 = ic_test_range[1]
    true_y0 = (r2 - r1) * torch.rand(100) + r1
    t = torch.arange(0., args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True

    # sample each parameter to build the tuples
    f_samples = random.choices(f_test, k=args.num_bundles_test)
    a0_samples = random.choices(a0_test, k=args.num_bundles_test)
    y0_samples = torch.tensor(random.choices(true_y0, k=args.num_bundles_test)).reshape(1, -1)

    # print(y0_samples.shape)
    diffeq_init = diffeq(a0_samples, f_samples)
    gt_generator = base_diffeq(diffeq_init)

    

    if not args.save:
        func.load_state_dict(torch.load(exp_name ))
    func.eval()

    h = func.h(t)

    hd = diff(h, t)
    h = h.detach()
    hd = hd.detach()

    gz_np = h.detach().numpy()
    T = np.linspace(0, 1, len(gz_np)) ** 2
    new_hiddens = scaler.fit_transform(gz_np)
    

    if args.plot_pca or args.plot_tsne:

        fig = plt.figure()
        ax = plt.axes(projection='3d')



        from sklearn.manifold import TSNE
        if args.plot_tsne:
            pca = PCA(n_components=args.plot_tsne)
        else:
            pca = PCA(n_components=3)

        comps = pca.fit_transform(new_hiddens)

        if plot_tsne:
            comps = TSNE(n_components=3).fit_transform(comps)
            comps = comps[comps[:, 0].argsort()]

        
        if comps.shape[1] >= 2:
            s = 10  # Segment length
            for i in range(0, len(gz_np) - s, s):

                if args.plot_tsne:
                    ax.plot3D(comps[i:i + s + 1, 0], comps[i:i + s + 1, 1], comps[i:i + s + 1, 2],
                              color=(0.1, 0.8, T[i]))
                else:

                    ax.plot3D(comps[i:i + s + 1, 0], comps[i:i + s + 1, 1], comps[i:i + s + 1, 2],
                              color=(0.1, 0.8, T[i]))
                plt.xlabel('comp1')
                plt.ylabel('comp2')
            

    s1 = time.time()

    wout, bias = wout_gen.get_wout(h, hd, y0_samples, t.detach(), a0_samples, f_samples)
    pred_y = h @ wout + bias

    s2 = time.time()
    print(f'all_ics:{s2 - s1}')

    s1 = time.time()
    true_ys = (gt_generator.get_solution(y0_samples, t.ravel())).reshape(-1, args.num_bundles_test)
    s2 = time.time()
    print(f'gt_ics:{s2 - s1}')

    # s1 = time.time()
    # true_y = estim_generator.get_solution(ics.reshape(-1, 1), t.ravel())
    # estim_ys = true_y.reshape(len(pred_y), ics.shape[1])
    # s2 = time.time()
    # print(f'estim_ics:{s2 - s1}')

    print(f'prediction_accuracy:{((pred_y - true_ys) ** 2).mean()} pm {((pred_y - true_ys) ** 2).std()}')
    #rint(f'estim_accuracy:{((estim_ys - true_ys) ** 2).mean()} pm {((estim_ys - true_ys) ** 2).std()}')

    # fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # # print(true_ys[0,:])
    # for i in range(0, args.num_bundles_test, 50):
    #     gt = true_ys.cpu().numpy()[:, i]
    #     preds = pred_y.cpu().numpy()[:, i]
    #     ax[0].plot(t.detach().cpu().numpy(), gt, c='blue', linestyle='dashed')
    #     ax[0].plot(t.detach().cpu().numpy(),  preds , c='orange')
    #     # plt.draw()

    # ax[1].plot(t.detach().cpu().numpy(), ((true_ys - pred_y) ** 2).mean(1).cpu().numpy(), c='green')
    # ax[1].set_xlabel('Time (s)')
    #plt.legend()
    #plt.show()

    # if args.save:

    #     fig.savefig(filename + "_pca")
    
    prediction_residuals = ((pred_y - true_ys) ** 2)
    #estimation_residuals = ((estim_ys - true_ys) ** 2)
    score = prediction_residuals.mean()
    return func.state_dict(), score, pred_y, true_ys, filename, loss_collector, h


if __name__ == "__main__":
    assert False, 'asdf'
    parser = parser_args.parse_args_bundles_('transfer demo')
    args = parser.parse_args()
    locals().update(args.__dict__)

    args = args.__dict__


    opt_hps = {'lr': 0.00021174227329280387, 'hidden_size': int(294.01715087890625), 'spikethreshold': 2.4561667442321777}

    for key in opt_hps.keys():
        try:
            del args[key]
        except:
            pass

    num_forces = 100

    #model, score, pred_y, true_y, filename, loss_collector = optimize(ic_tr_range, ic_te_range)

    # results["models"].append(model)
    # results["scores"].append(score)
    # results["pred_ys"].append(pred_y)
    # results["true_ys"].append(true_y)

    # with open(filename + '.pickle', 'wb') as f:
    #     pickle.dump(results, f)

    #evaluate_only = False
    #score, pred_y, true_y, filename = optimize(ic_tr_range, ic_te_range)
    if True:
        for n in [1]:#[1, 5, 10, 20, 50, 100, 500, 1000]:

            #assert ffnn_bias, 'asdf'
            #for ratio in [1, 2, 10]:#[1, 2, 10]:
            ratio = 1
            num_forces = max(n//ratio,1)
            #args.num_bundles = num_bundles = n

            print(f'NUM BUNDLES: {num_bundles}')
            #assert False, f'{num_bundles/num_forces}' 
        
            #if evaluate_only:
            results = {"models" : [], "scores" : [], "pred_ys" : [], "true_ys" : [], "loss" : [], "hs" : []}
            for i in range(1):
                
                model, score, pred_y, true_y, filename, loss_collector, h = optimize(ic_tr_range, ic_te_range, **opt_hps, args = args)
                results["models"].append(model)
                results["scores"].append(score)
                results["pred_ys"].append(pred_y)
                results["true_ys"].append(true_y)
                results["loss"].append(loss_collector)
                results["hs"].append(h)

            with open(filename + '.pickle', 'wb') as f:
                pickle.dump(results, f)

            # df2 = pd.read_pickle(filename + '.pickle')
            # print(df2)

#     