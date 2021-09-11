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

parser.add_argument('--tmax', type=float, default=3.)
parser.add_argument('--dt', type=int, default=0.1)
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--niters_test', type=int, default=15000)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--num_ics', type=int, default=1)
parser.add_argument('--num_test_ics', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--viz', action='store_false')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--evaluate_only', action='store_false')

args = parser.parse_args()
from torchdiffeq import odeint_adjoint as odeint


class diffeq(nn.Module):
    """
    defines the diffeq of interest
    """

    def __init__(self, a0, a1, f):
        super().__init__()
        self.a1 = a1
        self.a0 = a0
        self.f = f

    # return ydot
    def forward(self, t, states):
        # print(y.shape)
        y = states[:, 0].reshape(-1,1)
        yd = states[:,1].reshape(-1,1)
        ydd = (-self.a1(t) * yd -self.a0(t)*y + self.f(t)).reshape(-1,1)
        return torch.cat([yd,ydd],1)


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

class ODEFunc(nn.Module):
    """
    function to learn the outputs u(t) and hidden states h(t) s.t. u(t) = h(t)W_out
    """

    def __init__(self, hidden_dim,output_dim):
        super(ODEFunc, self).__init__()
        self.hdim = hidden_dim
        self.nl = SiLU()
        self.lin1 = nn.Linear(1, self.hdim)
        self.lin2 = nn.Linear(self.hdim, self.hdim)
        # self.lin3 = nn.Linear(self.hdim, self.hdim)

        self.lout = nn.Linear(self.hdim, output_dim, bias=False)

    def forward(self, t):
        x = self.lin1(t)
        x = self.nl(x)
        x = self.lin2(x)
        x = self.nl(x)
        # x = self.lin3(x)
        # x = self.nl(x)
        x = self.lout(x)
        return x

    def dot(self, t):
        outputs = self.forward(t)
        doutdt = [torch.autograd.grad(outputs[:, i].sum(), t, create_graph=True)[0] for i in range(outputs.shape[1])]
        return torch.cat(doutdt, 1)

    def ddot(self, t):
        outputs = self.dot(t)
        doutdt = [torch.autograd.grad(outputs[:, i].sum(), t, create_graph=True)[0] for i in range(outputs.shape[1])]
        return torch.cat(doutdt, 1)

    def wouts(self, x):
        return self.lout(x)

    def h(self, t):
        x = self.lin1(t)
        x = self.nl(x)
        x = self.lin2(x)
        x = self.nl(x)
        # x = self.lin3(x)
        # x = self.nl(x)

        return x

    def hdot(self, t):
        outputs = self.h(t)
        doutdt = [torch.autograd.grad(outputs[:, i].sum(), t, create_graph=True)[0] for i in range(outputs.shape[1])]
        return torch.cat(doutdt, 1)

    def hddot(self, t):
        outputs = self.hdot(t)
        doutdt = [torch.autograd.grad(outputs[:, i].sum(), t, create_graph=True)[0] for i in range(outputs.shape[1])]
        return torch.cat(doutdt, 1)


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

    def __init__(self, a0, a1, f, lambda_):
        super(Transformer_Analytic, self).__init__()

        self.a1 = a1
        self.a0 = a0
        self.f = f
        self.lambda_ = lambda_

    def get_wout(self, s, sd,sdd, y0, t):
        # y0 = torch.stack([y0 for _ in range(len(s))]).reshape(len(s), -1)
        a1 = self.a1(t).reshape(-1, 1)
        a0 = self.a0(t).reshape(-1, 1)
        f = self.f(t).reshape(-1, 1)

        DH = (sdd+a1 * sd + a0 * s)
        D0 = (-f).repeat_interleave(y0.shape[1]).reshape(-1, y0.shape[1])
        lambda_0 = self.lambda_

        h0m = s[0].reshape(-1, 1)
        h0d = sd[0].reshape(-1,1)
        W0 = torch.linalg.solve(DH.t() @ DH + lambda_0 + h0m @ h0m.t() + h0d @ h0d.t(), -DH.t() @ D0 + h0m @ (y0[0, :].reshape(1, -1))+ h0d @ (y0[1, :].reshape(1, -1)))
        return W0


def compute_h_hdot(func, batch_t):
    h = func.h(batch_t)
    hdot = func.hdot(batch_t)
    hddot = func.hddot(batch_t)

    return h,hdot,hddot


def compute_s_sdot(func, batch_t):
    s = func(batch_t)
    sd = func.dot(batch_t)
    sdd = func.ddot(batch_t)

    return s,sd,sdd

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
        for i in range(args.num_ics):
            ax_traj.plot(t.detach().cpu().numpy(), true_y.cpu().numpy()[:, i, 0],
                         'g-')
            ax_traj.plot(t.detach().cpu().numpy(), pred_y.cpu().numpy()[:, i, 0], '--', 'b--')
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
    a0 = lambda t: t ** 2
    a1 = lambda t: 0. + 0. * t
    f = lambda t: 0*torch.sin(t)

    diffeq_init = diffeq(a0, a1, f)
    gt_generator = base_diffeq(diffeq_init)
    estim_generator = estim_diffeq(diffeq_init)
    if args.num_ics == 1:
        true_y0 = torch.tensor([[1.,5.]])
    else:
        r1 = -5.
        r2 = 5.
        true_y0 = (r2-r1)*torch.rand(args.num_ics,2) + r1
    # print(true_y0.shape)
    t = torch.arange(0., args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True

    #use this quick test to find gt solutions and check training ICs
    #have a solution (don't blow up for dopri5 integrator)
    true_y = gt_generator.get_solution(true_y0, t.ravel())
    # print(true_y.shape)
    # instantiate wout with coefficients
    wout_gen = Transformer_Analytic(a0, a1, f, 0.0)
    func = ODEFunc(hidden_dim=NDIMZ,output_dim=args.num_ics)

    optimizer = optim.Adam(func.parameters(), lr=1e-3)

    loss_collector = []

    if not args.evaluate_only:

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
            pred_y, pred_ydot,pred_yddot = compute_s_sdot(func, tv)

            #enforce diffeq
            loss_diffeq = pred_yddot + (a1(tv.detach()).reshape(-1, 1)) * pred_ydot + (a0(tv.detach()).reshape(-1, 1)) * pred_y - f(tv.detach()).reshape(-1, 1)

            #enforce initial conditions
            loss_ics = torch.square(pred_y[0, :].ravel() - true_y0[:,0].ravel()) + torch.square(pred_ydot[0,:].ravel()-true_y0[:,1].ravel())
            # print(loss_ics.item())
            loss = torch.mean(torch.square(loss_diffeq)) + torch.mean(loss_ics)
            loss.backward()
            optimizer.step()
            loss_collector.append(torch.square(loss_diffeq).mean().item())
            if itr % args.test_freq == 0:
                func.eval()
                print(loss_collector[-1])
                s, _,_ = compute_s_sdot(func, t)
                pred_y = s.detach()
                pred_y = pred_y.reshape(-1, args.num_ics, 1)
                visualize(true_y.detach(), pred_y.detach(), loss_collector)
                ii += 1

        torch.save(func.state_dict(), 'func_ffnn_second_order')

    # with torch.no_grad():

    a0 = lambda t: t ** 2
    a1 = lambda t: 0. + 0. * t
    f = lambda t: 0 * torch.sin(t)

    diffeq_init = diffeq(a0, a1, f)
    gt_generator = base_diffeq(diffeq_init)
    estim_generator = estim_diffeq(diffeq_init)
    t = torch.arange(0., args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True
    wout_gen = Transformer_Analytic(a0, a1, f, 0.0)

    func.load_state_dict(torch.load('func_ffnn_second_order'))
    func.eval()

    r1 = -5.
    r2 = 5.

    h, hd,hdd = compute_h_hdot(func, t)
    h = h.detach()
    hd = hd.detach()
    hdd = hdd.detach()

    ics = (r2 - r1) * torch.rand(args.num_test_ics, 2) + r1

    loss_collector = []

    # y0 = ics.reshape(1, -1)
    s1 = time.time()
    wout = wout_gen.get_wout(h, hd,hdd, ics.t(), t.detach())
    pred_y = h @ wout
    s2 = time.time()
    print(f'all_ics:{s2 - s1}')

    s1 = time.time()
    true_y = gt_generator.get_solution(ics, t.ravel())
    true_ys = true_y[:,:,0].reshape(len(pred_y), ics.shape[0])
    s2 = time.time()
    print(f'gt_ics:{s2 - s1}')

    s1 = time.time()
    true_y = estim_generator.get_solution(ics, t.ravel())
    estim_ys = true_y[:,:,0].reshape(len(pred_y), ics.shape[0])
    s2 = time.time()
    print(f'estim_ics:{s2 - s1}')

    print(f'prediction_accuracy:{((pred_y - true_ys) ** 2).mean()} pm {((pred_y - true_ys) ** 2).std()}')
    print(f'estim_accuracy:{((estim_ys - true_ys) ** 2).mean()} pm {((estim_ys - true_ys) ** 2).std()}')

    plt.figure()
    # print(true_ys[0,:])
    for i in range(0,args.num_test_ics,50):
        plt.plot(t.detach().cpu().numpy(), true_ys.cpu().numpy()[:, i],c='blue',linestyle='dashed')
        plt.plot(t.detach().cpu().numpy(), pred_y.cpu().numpy()[:, i], c='orange')
        # plt.draw()
    plt.legend()
    plt.show()