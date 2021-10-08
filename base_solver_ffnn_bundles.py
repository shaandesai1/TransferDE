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



parser = argparse.ArgumentParser('transfer demo')

parser.add_argument('--tmax', type=float, default=3.)
parser.add_argument('--dt', type=int, default=0.1)
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--niters_test', type=int, default=15000)
parser.add_argument('--hidden_size', type=int, default=200)
parser.add_argument('--num_bundles', type=int, default=100)
parser.add_argument('--num_bundles_test', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--viz', action='store_false')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--evaluate_only', action='store_true')
args = parser.parse_args()
scaler = MinMaxScaler()

# print(args.evaluate_only==False)

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

    def __init__(self, hidden_dim, output_dim):
        super(ODEFunc, self).__init__()
        self.hdim = hidden_dim
        self.nl = nn.Tanh()
        self.lin1 = nn.Linear(1, self.hdim)
        self.lin2 = nn.Linear(self.hdim, self.hdim)
        self.lout = nn.Linear(self.hdim, output_dim, bias=True)

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
    r"""The derivative of a variable with respect to another.
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



def get_wout(s, sd, y0, t,a0s,fs):
    # y0 = torch.stack([y0 for _ in range(len(s))]).reshape(len(s), -1)

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

    W0 = torch.linalg.solve(DH.t() @ DH + h0m @ h0m.t(), -DH.t() @ D0 + h0m @ (y0[0, :].reshape(1, -1)))
    return W0
    # right_term = torch.einsum('ik,il->ilk', a0, s)
    # left_term = torch.einsum('ik,il->ilk', torch.ones_like(a0), sd)
    # DH = (left_term + right_term)
    # D0 = -f0
    #
    # DH = torch.einsum('ilk->kil',DH)
    # DHt = torch.einsum('kil->kli',DH)
    #
    # DHtDH = torch.einsum('kli,kil->kll',DHt,DH)
    # h0m = s[0].reshape(-1, 1)
    # W0 = torch.linalg.solve(DHtDH+ h0m @ h0m.t(), -DH.t() @ D0 + h0m @ (y0[0, :].reshape(1, -1)))
    # return W0

    # a0 = a0s(t).reshape(-1, 1)
    # a1 =1.
    # # f = fs(t).reshape(-1, 1)
    # f = torch.cat([f_(t) for f_ in fs], 1)
    #
    # DH = (a1 * sd + a0 * s)
    # D0 = (-f).repeat_interleave(y0.shape[1]).reshape(-1, y0.shape[1])
    # lambda_0 = self.lambda_
    #
    # h0m = s[0].reshape(-1, 1)
    # W0 = torch.linalg.solve(DH.t() @ DH + lambda_0 + h0m @ h0m.t(), -DH.t() @ D0 + h0m @ (y0[0, :].reshape(1, -1)))
    # return W0

import matplotlib.pyplot as plt

if args.viz:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


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


if __name__ == '__main__':

    ii = 0
    NDIMZ = args.hidden_size
    # define coefficients as lambda functions, used for gt and wout_analytic
    # training differential equation

    #need to sample tuple of (a1,f,IC)
    # each column of Wouts defines a solution thus, each tuple defines a solution too


    f_train = [lambda t: torch.cos(t),lambda t: torch.sin(t),lambda t: 0*t]
    a0_train = [lambda t: t,lambda t:t**2, lambda t: 0*t]
    r1 = -5.
    r2 = 5.
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
    func = ODEFunc(hidden_dim=NDIMZ, output_dim=args.num_bundles)

    optimizer = optim.Adam(func.parameters(), lr=1e-3)

    loss_collector = []

    best_residual = 1e-3

    if not args.evaluate_only:

        for itr in range(1, args.niters + 1):
            s1 = time.time()
            func.train()

            # add t0 to training times, including randomly generated ts
            t0 = torch.tensor([[0.]])
            t0.requires_grad = True
            tv = args.tmax * torch.rand(int(args.tmax / args.dt)).reshape(-1, 1)
            tv.requires_grad = True
            tv = torch.cat([t0, tv], 0)
            optimizer.zero_grad()

            # compute hwout,hdotwout
            pred_y = func(tv)
            pred_ydot = diff(pred_y, tv)

            # enforce diffeq
            loss_diffeq = pred_ydot - get_udot(tv,pred_y,a0_samples,f_samples)
            # loss_diffeq = (a1(tv.detach()).reshape(-1, 1)) * pred_ydot + (a0(tv.detach()).reshape(-1, 1)) * pred_y - f(
            #     tv.detach()).reshape(-1, 1)

            # enforce initial conditions
            loss_ics = pred_y[0, :].ravel() - y0_samples.ravel()

            loss = torch.mean(torch.square(loss_diffeq)) + torch.mean(torch.square(loss_ics))
            loss.backward()
            optimizer.step()
            loss_collector.append(torch.square(loss_diffeq).mean().item())

            print(time.time()-s1)
            if itr % args.test_freq == 0:
                func.eval()
                pred_y = func(t)
                pred_ydot = diff(pred_y,t)

                pred_y = pred_y.detach()
                pred_ydot = pred_ydot.detach()

                # pred_y = pred_y.reshape(-1, args.num_bundles)
                visualize(true_y.detach(), pred_y.detach(), loss_collector)
                ii += 1

                current_residual = torch.mean((pred_ydot - get_udot(t,pred_y,a0_samples,f_samples))**2)
                print(current_residual.item())
                # if current_residual < best_residual:
                #
                #     torch.save(func.state_dict(), 'func_ffnn_bundles')
                #     best_residual = current_residual
                #     print(itr,best_residual.item())
        # torch.save(func.state_dict(), 'func_ffnn_bundles')

    # with torch.no_grad():

    f_test = [lambda t: torch.sin(t)]
    # keep fixed to one list element - else need to do tensor math to compute Wout
    a0_test = [lambda t: t**3]
    r1 = -15.
    r2 = 15.
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


    func.load_state_dict(torch.load('func_ffnn_bundles'))
    func.eval()

    h = func.h(t)
    hd = diff(h, t)
    h = h.detach()
    hd = hd.detach()


    plt.figure()

    plt.plot(h)
    plt.show()



    gz_np = h.detach().numpy()
    T = np.linspace(0, 1, len(gz_np)) ** 2
    new_hiddens = scaler.fit_transform(gz_np)
    pca = PCA(n_components=3)
    pca_comps = pca.fit_transform(new_hiddens)

    fig = plt.figure()
    # ax = plt.axes(projection='3d')

    if pca_comps.shape[1] >= 2:
        s = 10  # Segment length
        for i in range(0, len(gz_np) - s, s):
            plt.plot(pca_comps[i:i + s + 1, 0], pca_comps[i:i + s + 1, 1])
        # s = 10  # Segment length
        # for i in range(0, len(gz_np) - s, s):
        #     ax.plot3D(pca_comps[i:i + s + 1, 0], pca_comps[i:i + s + 1, 1], pca_comps[i:i + s + 1, 2],
        #               color=(0.1, 0.8, T[i]))
        #     plt.xlabel('comp1')
        #     plt.ylabel('comp2')


    s1 = time.time()
    wout = get_wout(h, hd, y0_samples, t.detach(),a0_samples[0],f_samples)
    pred_y = h @ wout
    s2 = time.time()
    print(f'all_ics:{s2 - s1}')

    s1 = time.time()
    true_ys = (gt_generator.get_solution(y0_samples, t.ravel())).reshape(-1, args.num_bundles_test)
    s2 = time.time()
    print(f'gt_ics:{s2 - s1}')

    print(true_ys.shape,pred_y.shape)

    # s1 = time.time()
    # true_y = estim_generator.get_solution(ics.reshape(-1, 1), t.ravel())
    # estim_ys = true_y.reshape(len(pred_y), ics.shape[1])
    # s2 = time.time()
    # print(f'estim_ics:{s2 - s1}')

    print(f'prediction_accuracy:{((pred_y - true_ys) ** 2).mean()} pm {((pred_y - true_ys) ** 2).std()}')
    # print(f'estim_accuracy:{((estim_ys - true_ys) ** 2).mean()} pm {((estim_ys - true_ys) ** 2).std()}')

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # print(true_ys[0,:])
    for i in range(0, args.num_bundles_test, 50):
        ax[0].plot(t.detach().cpu().numpy(), true_ys.cpu().numpy()[:, i], c='blue', linestyle='dashed')
        ax[0].plot(t.detach().cpu().numpy(), pred_y.cpu().numpy()[:, i], c='orange')
        # plt.draw()

    ax[1].plot(t.detach().cpu().numpy(), ((true_ys - pred_y) ** 2).mean(1).cpu().numpy(), c='green')
    ax[1].set_xlabel('Time (s)')
    plt.legend()
    plt.show()
