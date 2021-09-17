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
        x = self.h(t)
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


def diff(u, t, order=1):
    # code adapted from neurodiffeq library
    # https://github.com/NeuroDiffGym/neurodiffeq/blob/master/neurodiffeq/neurodiffeq.py
    """The derivative of a variable with respect to another.
    """
    # ones = torch.ones_like(u)

    der = torch.cat([torch.autograd.grad(u[:, i].sum(), t, create_graph=True)[0] for i in range(u.shape[1])], 1)
    if der is None:
        print('derivative is None')
        return torch.zeros_like(t, requires_grad=True)
    else:
        der.requires_grad_()
    for i in range(1, order):

        der = torch.cat([torch.autograd.grad(der[:, i].sum(), t, create_graph=True)[0] for i in range(der.shape[1])], 1)
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

    def __init__(self, lambda_, calc_bias):
        super(Transformer_Analytic, self).__init__()

        self.calc_bias = calc_bias
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

        y0 = torch.stack([y0 for _ in range(len(s))]).reshape(len(s), -1)

        if self.calc_bias:
            ones_col = torch.ones_like(s[:,0]).view(-1,1)
            #states with bias
            s = torch.hstack((ones_col, s))

            #sd with zeros
            sd = torch.hstack((0 * ones_col, sd))


        

        a0 = a0s(t).reshape(-1, 1)
        a1 = torch.ones_like(a0)
        f = torch.cat([f_(t) for f_ in fs], 1)

        # a0 = torch.cat([a_(t) for a_ in a0s], 1)
        # f0 = torch.cat([f_(t) for f_ in fs], 1)
        # a1 = torch.ones_like(a0)
        # idms = torch.ones((s.shape[1],a0.shape[1]))
        D0 = -f

        DH = (a1*sd + a0 * s)
        h0m = s[0].reshape(-1, 1)


        LHS = DH.t() @ DH + h0m @ h0m.t()
        RHS = -DH.t() @ D0 + h0m @ (y0[0, :].reshape(1, -1))

        from sklearn.datasets import load_diabetes
        from sklearn.linear_model import RidgeCV
        #X, y = load_diabetes(return_X_y=True)
        clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(LHS, RHS)

        breakpoint()
        

        if self.lambda_:
            LHS = LHS + self.lambda_ * torch.eye(LHS.shape[0])
        W0 = torch.linalg.solve(LHS, RHS)

        #W0 = torch.linalg.solve( LHS , -DH.T @ D0 + h0m @ (y0[0, :].reshape(1, -1)))

        if self.calc_bias:
            weight = W0[1:]
            bias = W0[0]
            W0 = weight
        else:
            bias = 0
        return W0, bias



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


def visualize(true_y, pred_y, lst):
    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        for i in range(args.num_bundles):
            ax_traj.plot(t.detach().cpu().numpy(), true_y.cpu().numpy()[:, i],
                         'g-')
            ax_traj.plot(t.detach().cpu().numpy(), pred_y.cpu().numpy()[:, i], '--', 'b--')
        ax_phase.set_yscale('log')
        ax_phase.plot(np.arange(len(lst)), lst)

        ax_traj.legend()

        plt.draw()
        plt.pause(0.001)

