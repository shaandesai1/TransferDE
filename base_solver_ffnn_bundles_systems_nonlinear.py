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
import seaborn as sns

torch.manual_seed(33)

parser = argparse.ArgumentParser('transfer demo')

parser.add_argument('--tmax', type=float, default=3)
parser.add_argument('--dt', type=int, default=0.05)
parser.add_argument('--niters', type=int, default=15000)
parser.add_argument('--niters_test', type=int, default=5000)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--num_bundles', type=int, default=5)
parser.add_argument('--num_bundles_test', type=int, default=17)
parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--evaluate_only', action='store_false')
args = parser.parse_args()
scaler = MinMaxScaler()

# print(args.evaluate_only==False)

class diffeq(nn.Module):
    """
    defines the diffeq of interest
    """

    def __init__(self):
        super().__init__()

    # return ydot
    def forward(self, t, states):
        # y = y[:, 0]
        y = states[:, 0].reshape(1, -1)
        yd = states[:, 1].reshape(1, -1)
        ydd = get_udot(t,y)#(-self.a1(t) * yd - self.a0(t) * y + self.f(t)).reshape(-1, 1)
        return torch.cat([yd.reshape(-1,1), ydd.reshape(-1,1)], 1)

# get_udot(tv,pred_y,pred_ydot,a1_samples,a0_samples,f_samples)

def get_udot(t,y):

    ydd = -y - y**3
    return ydd

class SiLU(nn.Module):
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return torch.sin(input)


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
        self.nl = SiLU()
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


def get_ham(q,p):

    return (q**2)/2 + (q**4)/4 + (p**2)/2


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
        self.lin1 = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        return self.lin1(x)


# h, hd, hdd, true_y0, t.detach(), a1_train, a0_train, f_train)
def get_wout(s, sd,sdd, y0s, t,a1s,a0s,fs):
    # y0 = torch.stack([y0 for _ in range(len(s))]).reshape(len(s), -1)

    a0_batch = torch.cat([var_(t) for var_ in a0s], 1)
    a1_batch = torch.cat([var_(t) for var_ in a1s], 1)
    f_batch = torch.cat([var_(t) for var_ in fs], 1)
    WS = []
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
        # print(W0.shape)
        WS.append(W0)
    nWS = (torch.cat(WS)).reshape(f_batch.shape[1],-1)
    return nWS.t()

import matplotlib.pyplot as plt

if args.viz:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y,pred_ydot, lst,t):
    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        for i in range(args.num_bundles):
            ax_traj.plot(true_y.cpu().numpy()[:, i,0],true_y.cpu().numpy()[:, i,1],
                         'g-')
            ax_traj.plot(pred_y.cpu().numpy()[:,i], pred_ydot.cpu().numpy()[:, i])
        ax_phase.set_yscale('log')
        ax_phase.plot(np.arange(len(lst)), lst)

        # ax_traj.legend()

        plt.draw()
        plt.pause(0.001)


if __name__ == '__main__':

    ii = 0
    NDIMZ = args.hidden_size
    # define coefficients as lambda functions, used for gt and wout_analytic
    # training differential equation

    #need to sample tuple of (a1,f,IC)
    # each column of Wouts defines a solution thus, each tuple defines a solution too

    mu = 0
    r1 = 0.5
    r2 = 2.5
    true_y0 = (r2 - r1) * torch.rand(100,2) + r1
    true_y0[:,1] = 0.
    t = torch.arange(0., args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True


    diffeq_init = diffeq()
    gt_generator = base_diffeq(diffeq_init)
    true_y = gt_generator.get_solution(true_y0[:args.num_bundles],t.ravel())

    # use this quick test to find gt solutions and check training ICs
    # have a solution (don't blow up for dopri5 integrator)
    # true_y = gt_generator.get_solution(true_y0.reshape(-1, 1), t.ravel())

    # instantiate wout with coefficients
    func = ODEFunc(hidden_dim=NDIMZ, output_dim=args.num_bundles)

    optimizer = optim.Adam(func.parameters(), lr=1e-3,weight_decay=1e-5)

    loss_collector = []

    best_residual = 1e-1

    if not args.evaluate_only:

        for itr in range(1, args.niters + 1):
            func.train()

            # add t0 to training times, including randomly generated ts
            t0 = torch.tensor([[0.]])
            t0.requires_grad = True

            # tmax = torch.tensor([[4*np.pi]])
            # tmax.requires_grad = True

            tr = args.tmax * torch.rand(int(args.tmax / args.dt)).reshape(-1, 1)
            tr.requires_grad = True
            tv = torch.cat([t0, tr], 0)
            # tv.requires_grad = True
            optimizer.zero_grad()

            # compute hwout,hdotwout
            pred_y = func(tv)
            pred_ydot = diff(pred_y, tv)
            pred_yddot = diff(pred_ydot,tv)

            # enforce diffeq
            # get_udot(t, y, yd, a1, a0, f)
            loss_diffeq = pred_yddot - get_udot(tv,pred_y)
            # loss_diffeq = (a1(tv.detach()).reshape(-1, 1)) * pred_ydot + (a0(tv.detach()).reshape(-1, 1)) * pred_y - f(
            #     tv.detach()).reshape(-1, 1)
            # print(pred_y[0,:])
            # print(pred_y[-1,:])
            # enforce initial conditions
            loss_ics = torch.mean((pred_y[0, :].ravel() - true_y0[:args.num_bundles,0].ravel())**2) + torch.mean((pred_ydot[0,:].ravel()-true_y0[:args.num_bundles,1].ravel())**2)

            # print(loss_ics)

            loss = torch.mean(torch.square(loss_diffeq)) + torch.mean(loss_ics) +  torch.mean(((get_ham(pred_y,pred_ydot))-(get_ham(true_y0[:args.num_bundles,0].reshape(1,-1),true_y0[:args.num_bundles,1].reshape(1,-1))))**2)
            loss.backward()
            optimizer.step()
            loss_collector.append(torch.square(loss_diffeq).mean().item())
            # print(loss_collector[-1])
            if itr % args.test_freq == 0:
                func.eval()
                pred_y = func(t)
                pred_ydot = diff(pred_y,t)
                pred_yddot = diff(pred_ydot,t)

                pred_y = pred_y.detach()
                pred_ydot = pred_ydot.detach()
                pred_yddot = pred_yddot.detach()

                # loss_diffeq =

                # pred_y = pred_y.reshape(-1, args.num_bundles)
                visualize(true_y.detach(), pred_y.detach(),pred_ydot, loss_collector,t.detach())
                ii += 1

                # current_residual = (pred_yddot - get_udot(t, pred_y, pred_ydot, a1_samples, a0_samples, f_samples)) ** 2
                # print(current_residual.shape)
                # print((current_residual.mean(0)))


                current_residual = torch.mean((pred_yddot - get_udot(tv, pred_y))**2)
                # print(current_residual)
                if current_residual < best_residual:

                    torch.save(func.state_dict(), 'func_ffnn_bundles_nonlin')
                    best_residual = current_residual
                    print(itr,best_residual)
        # torch.save(func.state_dict(), 'func_ffnn_bundles')

    # with torch.no_grad():

    # f_test = [lambda t: torch.sin(t)]
    # # keep fixed to one list element - else need to do tensor math to compute Wout
    # a0_test = [lambda t: t**3]
    # r1 = -15.
    # r2 = 15.
    # true_y0 = (r2 - r1) * torch.rand(100) + r1

    func.load_state_dict(torch.load('func_ffnn_bundles_nonlin'))
    func.eval()

    t = torch.arange(0., args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True

    r1 = 0.5
    r2 = 2.
    true_y0 = (r2 - r1) * torch.rand(args.num_bundles_test, 2) + r1
    true_y0[:, 1] = 0.

    diffeq_init = diffeq()
    gt_generator = base_diffeq(diffeq_init)
    true_y = gt_generator.get_solution(true_y0, t.ravel())

    h = func.h(t)
    hd = diff(h, t)
    hdd = diff(hd, t)
    h = h.detach()
    hd = hd.detach()
    hdd = hdd.detach()




    new_net = Transformer_Learned(h.shape[1],args.num_bundles_test)

    optimizer = optim.Adam(new_net.parameters(), lr=1e-3)

    loss_collector = []

# best_residual = 1e-1

    s1 = time.time()
    for itr in range(1, args.niters_test + 1):
        new_net.train()

        optimizer.zero_grad()

        # compute hwout,hdotwout
        pred_y = new_net(h)
        pred_ydot = new_net(hd)
        pred_yddot = new_net(hdd)

        # enforce diffeq
        # get_udot(t, y, yd, a1, a0, f)
        loss_diffeq = pred_yddot - get_udot(t, pred_y)
        # loss_diffeq = (a1(tv.detach()).reshape(-1, 1)) * pred_ydot + (a0(tv.detach()).reshape(-1, 1)) * pred_y - f(
        #     tv.detach()).reshape(-1, 1)
        # print(pred_y[0,:])
        # print(pred_y[-1,:])
        # enforce initial conditions
        loss_ics = torch.mean((pred_y[0, :].ravel() - true_y0[:, 0].ravel()) ** 2) + torch.mean(
            (pred_ydot[0, :].ravel() - true_y0[:, 1].ravel()) ** 2)

        # print(loss_ics)
        loss = torch.mean(torch.square(loss_diffeq)) + torch.mean(loss_ics) + torch.mean(((get_ham(pred_y,
                                                                                                   pred_ydot)) - (
                                                                                              get_ham(true_y0[
                                                                                                      :,
                                                                                                      0].reshape(1, -1),
                                                                                                      true_y0[
                                                                                                      :,
                                                                                                      1].reshape(1,
                                                                                                                 -1)))) ** 2)

        loss.backward()
        optimizer.step()
        loss_collector.append(torch.square(loss_diffeq).mean().detach().numpy())

    print(f'time:{time.time()-s1}')
    print('error')
    print(np.mean((loss_diffeq**2).detach().numpy()),np.std((loss_diffeq**2).detach().numpy()))

    # sns.set_palette('deep')


    # sns.set_palette('deep')
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    import matplotlib.pyplot as plt

    sns.axes_style(style='ticks')
    sns.set_context("paper", font_scale=2.3,
                    rc={"font.size": 30, "axes.titlesize": 25, "axes.labelsize": 20, "axes.legendsize": 20,
                        'lines.linewidth': 2.5})
    sns.set_palette('deep')
    sns.set_color_codes(palette='deep')
    #
    # sns.axes_style(style='ticks')
    # sns.set_context("paper", font_scale=2,
    #                 rc={"font.size": 10, "axes.titlesize": 25, "axes.labelsize": 20, "axes.legendsize": 20,
    #                     'lines.linewidth': 2})
    # sns.set_palette('deep')


    with torch.no_grad():
        new_net.eval()

        pred_y = new_net(h).detach().numpy()
        pred_yd = new_net(hd).detach().numpy()

        # plt.figure()
        f, (a0,a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]},figsize=(6,8))
        for i in range(args.num_bundles_test):
            if i < 5:
                a0.plot(pred_y[:, i], pred_yd[:, i], label='pred', c='g',alpha=1,linewidth=3)
            else:
                a0.plot(pred_y[:,i],pred_yd[:,i],label='pred',c='b',linewidth=4)
                a0.plot(true_y[:,i,0],true_y[:,i,1],label='gt',linestyle='--',c='black',linewidth=2)
        a0.set_xlabel(r'$\psi$')
        a0.set_ylabel(r'$\dot{\psi}$')

        a1.set_yscale('log')
        a1.plot(np.arange(len(loss_diffeq))*args.dt,(loss_diffeq**2).mean(1),c='royalblue')
        a1.set_xlabel('Time (s)')
        a1.set_ylabel('Residuals')
        plt.tight_layout()
        # plt.show()
        plt.savefig('nl_osc.pdf',dpi=2400,bbox_inches='tight')