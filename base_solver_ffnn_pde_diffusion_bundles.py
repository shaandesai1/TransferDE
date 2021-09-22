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
parser.add_argument('--niters', type=int, default=5000)
parser.add_argument('--niters_test', type=int, default=15000)
parser.add_argument('--hidden_size', type=int, default=50)
parser.add_argument('--num_bundles', type=int, default=2)
parser.add_argument('--num_bundles_test', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--bs', type=int, default=100)

parser.add_argument('--viz', action='store_false')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--evaluate_only', action='store_false')

args = parser.parse_args()

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
        self.nl = nn.Tanh()
        self.lin1 = nn.Linear(2, self.hdim)
        self.lin2 = nn.Linear(self.hdim, self.hdim)
        # self.lin3 = nn.Linear(self.hdim, self.hdim)

        self.lout = nn.Linear(self.hdim, output_dim, bias=True)

    def hidden_states(self, t,x):
        inputs_ = torch.cat([t.reshape(-1,1),x.reshape(-1,1)],1)
        u = self.lin1(inputs_)
        u = self.nl(u)
        u = self.lin2(u)
        u = self.nl(u)
        return u

    def forward(self, t,x):
        u = self.hidden_states(t,x)
        u = self.lout(u)
        return u

    def wouts(self, x):
        return self.lout(x)

def diff(u, t, order=1):
    # code adapted from neurodiffeq library
    # https://github.com/NeuroDiffGym/neurodiffeq/blob/master/neurodiffeq/neurodiffeq.py
    r"""The derivative of a variable with respect to another.
    While there's no requirement for shapes, errors could occur in some cases.
    See `this issue <https://github.com/NeuroDiffGym/neurodiffeq/issues/63#issue-719436650>`_ for details
    """
    # ones = torch.ones_like(u)


    der = torch.cat([torch.autograd.grad(u[:, i].sum(), t, create_graph=True)[0] for i in range(u.shape[1])],1)
    if der is None:
        print('derivative is None')
        return torch.zeros_like(t, requires_grad=True)
    else:
        der.requires_grad_()
    for _ in range(1, order):

        der = torch.cat([torch.autograd.grad(der[:, i].sum(), t, create_graph=True)[0] for i in range(der.shape[1])],1)
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

    def __init__(self,f,lambda_val,lbc,rbc):
        super(Transformer_Analytic, self).__init__()
        self.f = f
        self.lambda_eqn = lambda_val
        self.lbc = lbc
        self.rbc = rbc
        # self.lambda_ = lambda_

    def get_wout(self, func,t,x,grid_t,grid_x):

        # print('enter wout')
        # s1=time.time()
        H = func.hidden_states(t,x)
        # s2 = time.time()
        # print(s2-s1)
        dHdt = diff(H,t)
        d2Hdx2 = diff(H,x,2)

        H = torch.cat([H, torch.ones(len(H), 1)], 1)
        dHdt = torch.cat([dHdt, torch.zeros(len(H), 1)], 1)
        d2Hdx2 = torch.cat([d2Hdx2, torch.zeros(len(H), 1)], 1)
        # s2 = time.time()
        # print(s2-s1)

        DH = (dHdt-self.lambda_eqn*d2Hdx2)

        H0 = func.hidden_states(grid_t[:,0].reshape(-1,1),grid_x[:,0].reshape(-1,1))
        H0 = torch.cat([H0,torch.ones(len(H0),1)],1)
        HL = func.hidden_states(grid_t[0,:].reshape(-1,1),grid_x[0,:].reshape(-1,1))
        HL = torch.cat([HL, torch.ones(len(H0),1)],1)
        HR = func.hidden_states(grid_t[-1, :].reshape(-1, 1), grid_x[-1, :].reshape(-1, 1))
        HR = torch.cat([HR, torch.ones(len(H0),1)],1)

        F = self.f(grid_x[:,0]).reshape(-1,1)
        BL = self.lbc(grid_t[0,:]).reshape(-1,1)
        BR = self.rbc(grid_t[0,:]).reshape(-1,1)

        W0 = torch.linalg.solve(DH.t() @ DH  +H0.t()@H0 + HL.t()@HL+HR.t()@HR,H0.t()@F + HL.t()@BL + HR.t()@BR)
        return W0



if args.viz:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(u,t,x,grid_t,grid_x,lst):
    if args.viz:
        ax_traj.cla()
        ax_phase.cla()
        # print(grid_t.shape)
        ax_traj.contourf(x,t,u[:,0].reshape(len(x),len(t)).t())

        u_true = torch.sin(grid_x)*torch.exp(-grid_t)

        ax_phase.contourf(x,t,u_true.reshape(len(x),len(t)).t())

        ax_vecfield.contourf(x, t, (u_true.reshape(len(x), len(t)).t()-u[:,0].reshape(len(x),len(t)).t())**2)

        # ax_vecfield.set_yscale('log')
        # ax_vecfield.plot(np.arange(len(lst)), lst)

        ax_traj.legend()

        plt.draw()
        plt.pause(0.001)


if __name__ == '__main__':

    ii = 0
    NDIMZ = args.hidden_size
    # define coefficients as lambda functions, used for gt and wout_analytic
    # training differential equation



    xl = 0
    xr = torch.tensor(np.pi)
    t0 = 0
    tmax = args.tmax


    x_evals = torch.linspace(xl,xr,50)
    y_evals = torch.linspace(t0,tmax,50)
    grid_x, grid_t = torch.meshgrid(x_evals, y_evals)
    grid_x.requires_grad = True
    grid_t.requires_grad = True

    # print(grid_t[:,0], grid_x[:,0])

    grid_x = grid_x.ravel()
    grid_t = grid_t.ravel()


    # left BC
    bc_left = torch.ones(50)*xl
    bc_right = torch.ones(50)*xr
    ic_t0 = torch.cat([torch.sin(x_evals).reshape(-1,1),torch.sin(2*x_evals).reshape(-1,1)],1)

    bc_left.requires_grad=True
    bc_right.requires_grad=True
    ic_t0.requires_grad=True

    # wout_gen = Transformer_Analytic()
    func = ODEFunc(hidden_dim=NDIMZ,output_dim=args.num_bundles)

    optimizer = optim.Adam(func.parameters(), lr=1e-3)

    loss_collector = []

    if not args.evaluate_only:

        for itr in range(1, args.niters + 1):
            func.train()
            indices = torch.tensor(np.random.choice(len(grid_x),1000))
            x_tr = (grid_x[indices]).reshape(-1,1)
            t_tr = (grid_t[indices]).reshape(-1,1)

            # add t0 to training times, including randomly generated ts
            optimizer.zero_grad()

            u = func(t_tr,x_tr)
            dudt = diff(u,t_tr)
            d2udx2 = diff(u, x_tr, 2)

            lambda_0 = 1.
            #enforce diffeq
            loss_diffeq = torch.mean((dudt - lambda_0*d2udx2)**2)

            u_t0 = torch.mean((func(torch.zeros(len(x_evals)).reshape(-1,1),x_evals.reshape(-1,1)) - ic_t0)**2)
            u_left = torch.mean((func(y_evals.reshape(-1,1),bc_left.reshape(-1,1)) - 0) ** 2)
            u_right = torch.mean((func(y_evals.reshape(-1, 1), bc_right.reshape(-1,1)) - 0) ** 2)

            #enforce initial conditions
            loss_ics = u_t0 + u_left + u_right
            loss = loss_diffeq + loss_ics
            loss.backward()
            optimizer.step()
            loss_collector.append(loss_diffeq.item())
            if itr % args.test_freq == 0:
                with torch.no_grad():
                    func.eval()
                    print(f'diffeq: {loss_collector[-1]}, bcs: {loss_ics.item()}')
                    u_eval = func(grid_t.reshape(-1,1),grid_x.reshape(-1,1))
                    # s, _,_ = compute_s_sdot(func, t)
                    # pred_y = s.detach()
                    # pred_y = pred_y.reshape(-1, args.num_ics, 1)
                    visualize(u_eval,y_evals,x_evals,grid_t,grid_x, loss_collector)
                # ii += 1

        torch.save(func.state_dict(), 'func_ffnn_diffusion')

    # with torch.no_grad():

    f = lambda t: torch.sin(2*t)
    lbc = lambda t: 0*t
    rbc = lambda t: 0*t
    lambda_val = 1

    wout_gen = Transformer_Analytic(f,lambda_val,lbc,rbc)

    func.load_state_dict(torch.load('func_ffnn_diffusion'))
    func.eval()

    x_evals = torch.linspace(xl, xr, 50)
    y_evals = torch.linspace(t0, tmax, 50)
    x_evals.requires_grad = True
    y_evals.requires_grad = True
    grid_x, grid_t = torch.meshgrid(x_evals, y_evals)
    # grid_x.requires_grad = True
    # grid_t.requires_grad = True

    # print(grid_t[:,0], grid_x[:,0])

    grid_xx = grid_x.ravel()
    grid_tt = grid_t.ravel()

    WOUT = wout_gen.get_wout(func,grid_tt.reshape(-1,1),grid_xx.reshape(-1,1),grid_t,grid_x,)

    H = func.hidden_states(grid_tt.reshape(-1,1),grid_xx.reshape(-1,1))
    H = torch.cat([H,torch.ones(len(H),1)],1)
    with torch.no_grad():
        out_pred = (H@WOUT)
        fig,ax = plt.subplots(1,3,figsize=(20,7))

        ax[0].contourf(x_evals, y_evals, out_pred.reshape(len(x_evals), len(y_evals)).t())
        ax[0].set_title('predicted solution')


        u_true = torch.sin(2*grid_xx) * torch.exp(-4*lambda_val*grid_tt)

        ax[1].contourf(x_evals, y_evals, u_true.reshape(len(x_evals), len(y_evals)).t())
        ax[1].set_title('true solution')

        pc = ax[2].contourf(x_evals, y_evals, (u_true.reshape(len(x_evals), len(y_evals)).t() - out_pred.reshape(len(x_evals), len(y_evals)).t()) ** 2)
        ax[2].set_title('residuals')
        fig.colorbar(pc, ax=ax[2])

        plt.show()