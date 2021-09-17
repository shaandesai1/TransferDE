"""
base solver for transfer ode (first order methods)
"""
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import numpy as np
import time

parser = argparse.ArgumentParser('transfer demo')

parser.add_argument('--tmax', type=float, default=2)
parser.add_argument('--dt', type=int, default=0.01)
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--niters_test', type=int, default=15000)
parser.add_argument('--hidden_size', type=int, default=1000)
parser.add_argument('--num_ics', type=int, default=1)
parser.add_argument('--num_test_ics', type=int, default=100)
parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--viz', action='store_false')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--evaluate_only', action='store_true')

args = parser.parse_args()
from torchdiffeq import odeint_adjoint as odeint


class diffeq(nn.Module):
    """
    defines the diffeq of interest
    """

    def __init__(self,sigma,rho,beta):
        super().__init__()
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    # return ydot
    def forward(self,t, X):
        # print(X.shape)
        u, v, w = X[:,0],X[:,1],X[:,2]
        up = -self.sigma * (u - v)
        vp = self.rho * u - v - u * w
        wp = -self.beta * w + u * v
        # print(up)
        return torch.cat([up.reshape(-1,1), vp.reshape(-1,1), wp.reshape(-1,1)],1)


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

def diff(u, t, order=1):
    # code adapted from neurodiffeq library
    # https://github.com/NeuroDiffGym/neurodiffeq/blob/master/neurodiffeq/neurodiffeq.py
    r"""The derivative of a variable with respect to another.
    While there's no requirement for shapes, errors could occur in some cases.
    See `this issue <https://github.com/NeuroDiffGym/neurodiffeq/issues/63#issue-719436650>`_ for details
    :param u: The :math:`u` in :math:`\displaystyle\frac{\partial u}{\partial t}`.
    :type u: `torch.Tensor`
    :param t: The :math:`t` in :math:`\displaystyle\frac{\partial u}{\partial t}`.
    :type t: `torch.Tensor`
    :param order: The order of the derivative, defaults to 1.
    :type order: int
    :returns: The derivative evaluated at ``t``.
    :rtype: `torch.Tensor`
    """
    # ones = torch.ones_like(u)


    der = torch.cat([torch.autograd.grad(u[:, i].sum(), t, create_graph=True)[0] for i in range(u.shape[1])],1)
    if der is None:
        print('derivative is None')
        return torch.zeros_like(t, requires_grad=True)
    else:
        der.requires_grad_()
    for i in range(1, order):

        der = torch.cat([torch.autograd.grad(der[:, i].sum(), t, create_graph=True)[0] for i in range(der.shape[1])],1)
        # print()
        if der is None:
            print('derivative is None')
            return torch.zeros_like(t, requires_grad=True)
        else:
            der.requires_grad_()
    return der


class ODEFunc(nn.Module):
    """
    function to learn the outputs u(t) and hidden states h(t) s.t. u(t) = h(t)W_out
    """

    def __init__(self, hidden_dim,output_dim):
        super(ODEFunc, self).__init__()
        self.hdim = hidden_dim
        self.nl = SiLU()#nn.Tanh()
        self.lin1 = nn.Linear(1, self.hdim)
        self.lin2 = nn.Linear(self.hdim, self.hdim)
        self.lout = nn.Linear(self.hdim, output_dim, bias=False)

    def h(self, t):
        x = self.lin1(t)
        x = self.nl(x)
        x = self.lin2(x)
        x = self.nl(x)
        return x

    def forward(self, t):
        x = self.h(t)
        x = self.lout(x)
        return x

    def wouts(self, x):
        return self.lout(x)



class Transformer_Learned(nn.Module):
    """
    returns Wout learnable, only need hidden and output dims
    """

    def __init__(self, input_dims, output_dims):
        super(Transformer_Learned, self).__init__()
        self.lin1 = nn.Linear(args.hidden_size, output_dims)

    def forward(self, x):
        return self.lin1(x)

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


if args.viz:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y,lst):
    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        # for i in range(3):
        ax_traj.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1],
                     'g-')
        ax_traj.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], '--', 'b--')
        ax_phase.set_yscale('log')
        ax_phase.plot(np.arange(len(lst)), lst)

        ax_traj.legend()

        plt.draw()
        plt.pause(0.001)


if __name__ == '__main__':

    ii = 0
    NDIMZ = args.hidden_size

    sigma = 10
    beta = 2.667
    rho = 28
    diffeq_init = diffeq(sigma,rho,beta)
    gt_generator = base_diffeq(diffeq_init)
    estim_generator = estim_diffeq(diffeq_init)
    if args.num_ics == 1:
        true_y0 = torch.tensor([[0.1,1.,1.05]]).reshape(1,3)
    else:
        r1 = -5.
        r2 = 5.
        true_y0 = (r2-r1)*torch.rand(args.num_ics) + r1
    t = torch.arange(0., args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True

    #use this quick test to find gt solutions and check training ICs
    #have a solution (don't blow up for dopri5 integrator)
    true_y = gt_generator.get_solution(true_y0.reshape(1,3), t.ravel())

    func = ODEFunc(hidden_dim=NDIMZ,output_dim=3*args.num_ics)

    optimizer = optim.Adam(func.parameters(), lr=1e-3)

    loss_collector = []

    if not args.evaluate_only:
        diffeq_init.eval()
        for itr in range(1, args.niters + 1):
            func.train()
            optimizer.zero_grad()


            # add t0 to training times, including randomly generated ts
            t0 = torch.tensor([[0.]])
            t0.requires_grad = True
            tv = args.tmax * torch.rand(100).reshape(-1, 1)
            tv.requires_grad = True
            tv = torch.cat([t0, tv], 0)


            # compute hwout,hdotwout

            pred_y = func(tv)
            pred_ydot = diff(pred_y,tv)

            #enforce diffeq
            loss_diffeq = (pred_ydot-(diffeq_init(0,pred_y)))**2

            #enforce initial conditions
            loss_ics = (pred_y[0,:].ravel()-true_y0.ravel())**2

            loss = 0.1*torch.mean(loss_diffeq) + torch.mean(loss_ics)

            loss.backward()
            optimizer.step()
            loss_collector.append(torch.square(loss_diffeq).mean().item())
            if itr % args.test_freq == 0:
                func.eval()
                print(f'diff loss:{loss_collector[-1]}, ic loss: {loss_ics.mean().item()}')
                pred_y = func(t).detach()
                pred_y = pred_y.reshape(-1, 1, 3)
                visualize(true_y.detach(), pred_y.detach(), loss_collector)
                ii += 1

        torch.save(func.state_dict(), 'func_ffnn_lorenz')

    # with torch.no_grad():

    sigma = 10
    beta = 2.667
    rho = 28

    diffeq_init = diffeq(sigma,rho,beta)
    gt_generator = base_diffeq(diffeq_init)
    estim_generator = estim_diffeq(diffeq_init)
    t = torch.arange(0., args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True

    func.load_state_dict(torch.load('func_ffnn_lorenz'))
    func.eval()

    wout_gen = Transformer_Learned(NDIMZ,3)

    optimizer = torch.optim.Adam(wout_gen.parameters(),lr=1e-3)

    h = func.h(t)
    hd = diff(h,t)
    h = h.detach()
    hd = hd.detach()



    r1 = -5.
    r2 = 5.

    ics = torch.tensor([1,1,1.05]).reshape(1,-1)#torch.linspace(r1, r2, args.num_test_ics).reshape(1, -1)
    # print(ics)
    loss_collector = []

    y0 = ics.reshape(1, -1)
    s1 = time.time()

    for itr in range(args.niters_test):
        optimizer.zero_grad()

        t0 = torch.tensor([[0.]])
        t0.requires_grad = True
        tv = args.tmax * torch.rand(int(args.tmax / args.dt)).reshape(-1, 1)
        tv.requires_grad = True
        tv = torch.cat([t0, tv], 0)

        pred_y = wout_gen(h)
        pred_ydot = wout_gen(hd)

        # enforce diffeq
        loss_diffeq = (pred_ydot - (diffeq_init(0,pred_y))) ** 2

        # enforce initial conditions
        loss_ics = (pred_y[0, :].ravel() - ics.ravel()) ** 2

        loss = torch.mean(loss_diffeq) + torch.mean(loss_ics)
        loss.backward()
        optimizer.step()
        loss_collector.append(torch.square(loss_diffeq).mean().item())
        if itr % args.test_freq == 0:
            print(loss_collector[-1])
    with torch.no_grad():
        pred_y = wout_gen(h)
    s2 = time.time()
    print(f'all_ics:{s2 - s1}')

    s1 = time.time()
    true_y = gt_generator.get_solution(ics.reshape(-1, 3), t.ravel())
    true_ys = true_y.reshape(len(pred_y), ics.shape[1])
    s2 = time.time()
    print(f'gt_ics:{s2 - s1}')

    s1 = time.time()
    true_y = estim_generator.get_solution(ics.reshape(-1, 3), t.ravel())
    estim_ys = true_y.reshape(len(pred_y), ics.shape[1])
    s2 = time.time()
    print(f'estim_ics:{s2 - s1}')

    print(f'prediction_accuracy:{((pred_y - true_ys) ** 2).mean()} pm {((pred_y - true_ys) ** 2).std()}')
    print(f'estim_accuracy:{((estim_ys - true_ys) ** 2).mean()} pm {((estim_ys - true_ys) ** 2).std()}')

    fig,ax = plt.subplots(2,1,figsize=(10,8),sharex=True)
    # print(true_ys[0,:])
    for i in range(0,args.num_test_ics,50):
        ax[0].plot(t.detach().cpu().numpy(), true_ys.cpu().numpy()[:, i],c='blue',linestyle='dashed')
        ax[0].plot(t.detach().cpu().numpy(), pred_y.cpu().numpy()[:, i], c='orange')
        # plt.draw()

    ax[1].plot(t.detach().cpu().numpy(),((true_ys-pred_y)**2).mean(1).cpu().numpy(),c='green')
    ax[1].set_xlabel('Time (s)')
    plt.legend()
    plt.show()