"""
base solver for transfer ode (first order methods)
"""
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import numpy as np
# import time
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser('transfer demo')

parser.add_argument('--tmax', type=float, default=3.)
parser.add_argument('--dt', type=int, default=0.1)
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--niters_test', type=int, default=15000)
parser.add_argument('--hidden_size', type=int, default=50)
parser.add_argument('--num_ics', type=int, default=3)
parser.add_argument('--num_test_ics', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--bs', type=int, default=100)

parser.add_argument('--viz', action='store_true')
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
        self.nl1 = nn.Tanh()
        self.nl2 = SiLU()
        self.lin1 = nn.Linear(2, self.hdim)
        self.lin2 = nn.Linear(self.hdim, self.hdim)
        # self.weight_1 = nn.Parameter(torch.zeros(1))
        self.weight_2 = nn.Parameter(torch.zeros(1))
        # self.lin3 = nn.Linear(self.hdim, self.hdim)

        self.lout = nn.Linear(self.hdim, output_dim, bias=True)

    def hidden_states(self, t,x):
        inputs_ = torch.cat([t.reshape(-1,1),x.reshape(-1,1)],1)
        u = self.lin1(inputs_)
        u = self.nl1(u)#self.weight_1*self.nl1(u) + (1.-self.weight_1)*self.nl2(u)
        u = self.lin2(u)
        u = self.weight_2*self.nl1(u) + (1.-self.weight_2)*self.nl2(u)
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

    def __init__(self):
        super(Transformer_Analytic, self).__init__()


    def append_ones(self,var,type='ones'):

        if type == 'ones':
            return torch.cat([var,torch.ones(len(var),1)],1)

        else:
            return torch.cat([var,torch.zeros(len(var),1)],1)

    def get_wout(self, func,t,x,grid_t,grid_x,sigma,p0):

        H = torch.cat([func.hidden_states(t,x),torch.ones(len(t),1)],1)
        dHdt = diff(H,t,1)#torch.cat([diff(H,t,1),torch.zeros(len(t),1)],1)
        d2Hdx2 =diff(H,x,2) #torch.cat([diff(H,x,2),torch.zeros(len(t),1)],1)

        Amatrix = get_block_matrix(torch.tensor(1.))


        HHt = torch.block_diag(dHdt, dHdt)
        HHxx = torch.block_diag(d2Hdx2, d2Hdx2)
        HH = torch.block_diag(H,H)

        # print(HHt.shape,HHxx.shape,HH.shape)


        Amatrixhat = torch.zeros((HH.shape[0], HH.shape[0]))
        # print(Amatrix.shape)
        for i in range(Amatrix.shape[0]):
            for j in range(Amatrix.shape[1]):
                # print(Amatrix[i,j])
                Amatrixhat[i * H.shape[0]:(i + 1) * H.shape[0], j * H.shape[0]:(j + 1) * H.shape[0]] = torch.eye(
                    H.shape[0], H.shape[0]) * Amatrix[i, j]

        # DH = hddothat + Amatrixhat @ hhat

        DH = (HHt-Amatrixhat@HHxx)

        H0 = func.hidden_states(grid_t[:, 0].reshape(-1, 1), grid_x[:, 0].reshape(-1, 1))
        H0 = self.append_ones(H0)
        HH0 = torch.block_diag(H0,H0)

        lbc = grid_x[0, :].reshape(-1, 1)
        HL = func.hidden_states(grid_t[0, :].reshape(-1, 1),lbc)
        HL = self.append_ones(HL)
        HHL = torch.block_diag(HL, HL)

        HLd = diff(HL,lbc)
        #HLd = self.append_ones(HLd,'zeros')
        HHLd = torch.block_diag(HLd, HLd)

        rbc = grid_x[-1, :].reshape(-1, 1)
        HR = func.hidden_states(grid_t[-1, :].reshape(-1, 1), rbc)
        HR = self.append_ones(HR)
        HHR = torch.block_diag(HR, HR)

        HRd = diff(HR, rbc)
        #HRd = self.append_ones(HRd,'zeros')
        HHRd = torch.block_diag(HRd, HRd)

        IC = get_ic(grid_t[:, 0].reshape(-1, 1), grid_x[:, 0].reshape(-1, 1), sigma=sigma, x0=0., p0=p0)
        new_IC = torch.cat([IC[:,0].ravel(),IC[:,1].ravel()]).reshape(-1,1)

        # print(DH.shape,HH0.shape,HHL.shape,HHR.shape,HHLd.shape,HHRd.shape,new_IC.shape)

        W0 = torch.linalg.solve(DH.t() @ DH + HH0.t() @ HH0 + (HHL-HHR).t()@(HHL-HHR) + (HHLd-HHRd).t()@(HHLd-HHRd),HH0.t()@new_IC)

        nwout = torch.cat([W0[:args.hidden_size + 1, 0].reshape(-1, 1), W0[args.hidden_size + 1:, 0].reshape(-1, 1)], 1)

        return nwout,dHdt,d2Hdx2


if args.viz:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 6), facecolor='white')
    ax_traj = fig.add_subplot(221, frameon=False)
    ax_phase = fig.add_subplot(222, frameon=False)
    ax_vecfield = fig.add_subplot(223, frameon=False)
    ax_vecfield2 = fig.add_subplot(224, frameon=False)
    plt.show(block=False)


def visualize(func,u,t,x,grid_t,grid_x,lst):
    if args.viz:
        ax_traj.cla()
        ax_phase.cla()
        ax_vecfield.cla()
        ax_vecfield2.cla()
        # ax_traj.contourf(x,t,u[:,0].reshape(len(x),len(t)).t())
        # u_true = torch.sin(grid_x)*torch.exp(-grid_t)
        ax_phase.contourf(x, t, (u[:, 0]**2 + u[:,1]**2).reshape(len(x), len(t)).t())


        unew = func(torch.ones(500)*0.5,torch.linspace(-10.,10.,500))

        # unew1 = func(torch.ones(50) * 2., torch.linspace(-10., 10., 50))
        ax_vecfield.plot(unew[:, 0], unew[:, 1])

        ax_vecfield2.plot(unew[:,0])
        ax_vecfield2.plot(unew[:,1])

        # ax_vecfield2.scatter(unew[:,0],unew[:,1])

        # ax_vecfield2.scatter(unew1[:, 0], unew1[:, 1])

        # ax_vecfield.contourf(x, t, (u_true.reshape(len(x), len(t)).t()-u.reshape(len(x),len(t)).t())**2)
        ax_traj.legend()
        plt.draw()
        plt.pause(0.001)


def get_ic(t,x,sigma=0.5,x0=0.,p0=1.):
    t = t.reshape(-1,1)
    x = x.reshape(-1,1)

    hbar = 1.

    sigma = torch.tensor(sigma)

    c1= 1./np.pi**(1./4.)
    c2 = 1./torch.sqrt(sigma)
    c3 = torch.exp(-((x-x0)**2)/(2*sigma**2))
    c4 = torch.cos(p0*x/hbar)
    c5 = torch.sin(p0*x/hbar)

    real = c1*c2*c3*c4
    img = c1*c2*c3*c5

    return torch.cat([real,img],1)


def get_block_matrix(m1):
    Amatrix = []
    hbar = 1.
    # print(m1.shape)
    m1 = m1.reshape(1,-1)
    # m2 = m2.reshape(1,-1)
    # print(m1[0,0])
    for i in range(m1.shape[1]):
        Amatrix.append(torch.tensor([[0.,-hbar/(2.*m1[0,i])],[hbar/(2.*m1[0,i]),0]]))

    # print(torch.block_diag(*Amatrix).shape)
    return torch.block_diag(*Amatrix)



def get_transform(x):
    hbar = 1.
    m = 1.
    Amatrix = torch.tensor([[0.,-hbar/(2.*m)],[hbar/(2.*m),0]])

    output = Amatrix @ x.t()
    return output.t()


if __name__ == '__main__':

    ii = 0
    NDIMZ = args.hidden_size
    xl = -10.
    xr = 10.#torch.tensor(np.pi)
    t0 = 0.
    tmax = 1.#torch.tensor(np.pi)
    x_evals = torch.linspace(xl,xr,100)
    y_evals = torch.linspace(t0,tmax,100)
    grid_x, grid_t = torch.meshgrid(x_evals, y_evals)
    grid_x.requires_grad = True
    grid_t.requires_grad = True

    grid_x = grid_x.ravel()
    grid_t = grid_t.ravel()

    bc_left = torch.ones(100)*xl
    bc_right = torch.ones(100)*xr
    ic_t0 = torch.ones(100)*t0
    ic_tmax = torch.ones(100)*tmax


    ms_diffeq = torch.ones(args.num_ics)
    DIFF_MATRIX = get_block_matrix(ms_diffeq)

    bc_left.requires_grad=True
    bc_right.requires_grad=True
    ic_t0.requires_grad=True
    ic_tmax.requires_grad=True

    func = ODEFunc(hidden_dim=NDIMZ,output_dim=2*args.num_ics)

    optimizer = optim.Adam(func.parameters(), lr=1e-2)

    loss_collector = []
    best_residual = 1e-1
    if not args.evaluate_only:

        for itr in range(1, args.niters + 1):
            func.train()
            indices = torch.tensor(np.random.choice(len(grid_x),1000))
            x_tr = (grid_x[indices]).reshape(-1,1) + 0.1*torch.rand(indices)
            t_tr = (grid_t[indices]).reshape(-1,1) + 0.1*torch.rand(indices)

            # add t0 to training times, including randomly generated ts
            optimizer.zero_grad()

            u = func(t_tr,x_tr)
            dudt = diff(u,t_tr,1)
            d2udx2 = diff(u, x_tr, 2)
            loss_diffeq = torch.mean((dudt - d2udx2@DIFF_MATRIX.t())**2)

            init_force = torch.cat([get_ic(ic_t0,x_evals,sigma=0.5,p0=1.),get_ic(ic_t0,x_evals,sigma=0.6,p0=2.),get_ic(ic_t0,x_evals,sigma=0.7,p0=3.),],1)
            u_t0 = torch.mean((func(ic_t0, x_evals.reshape(-1, 1)) - init_force) ** 2)
            xleft  =bc_left.reshape(-1, 1)
            yleft = func(y_evals.reshape(-1, 1),xleft )
            yleft_x = diff(yleft,xleft)

            xright = bc_right.reshape(-1, 1)
            yright = func(y_evals.reshape(-1, 1), xright)
            yright_x = diff(yright, xright)

            u_period = torch.mean((yleft - yright) ** 2) + torch.mean((yleft_x - yright_x) ** 2)

            loss_ics = u_period + u_t0
            loss = loss_diffeq + loss_ics
            loss.backward()
            optimizer.step()
            loss_collector.append(loss_diffeq.item())
            if itr % args.test_freq == 0:
                # with torch.no_grad():
                func.eval()
                print(f'diffeq: {loss_collector[-1]}, bcs: {loss_ics.item()}')
                grid_t_n,grid_x_n=grid_t.reshape(-1, 1), grid_x.reshape(-1, 1)
                u = func(grid_t_n,grid_x_n)
                dudt = diff(u, grid_t_n, 1)
                d2udx2 = diff(u, grid_x_n, 2)

                u.detach_()
                dudt.detach_()
                d2udx2.detach_()
                loss_diffeq = torch.mean((dudt - d2udx2 @ DIFF_MATRIX.t()) ** 2)

                visualize(func,u,y_evals,x_evals,grid_t,grid_x, loss_collector)
                if loss_diffeq.item() < best_residual:
                    torch.save(func.state_dict(), 'func_ffnn_schroed')
                    best_residual = loss_diffeq.item()

    func.load_state_dict(torch.load('func_ffnn_schroed'))
    func.eval()

    x_evals = torch.linspace(xl, xr, 100)
    y_evals = torch.linspace(t0, tmax, 100)
    x_evals.requires_grad = True
    y_evals.requires_grad = True
    grid_x, grid_t = torch.meshgrid(x_evals, y_evals)
    # grid_x.requires_grad = True
    # grid_t.requires_grad = True

    # print(grid_t[:,0], grid_x[:,0])
    wout_gen = Transformer_Analytic()
    grid_xx = grid_x.ravel()
    grid_tt = grid_t.ravel()

    WOUT,Ht,Hxx = wout_gen.get_wout(func,grid_tt.reshape(-1,1),grid_xx.reshape(-1,1),grid_t,grid_x,0.8,2.)

    H = torch.cat([func.hidden_states(grid_tt.reshape(-1,1),grid_xx.reshape(-1,1)),torch.ones(len(Ht),1)],1)

    print(WOUT.shape,H.shape,Ht.shape,Hxx.shape)

    with torch.no_grad():
        out_pred = (H@WOUT)
        # print((out_pred))
        loss_test = torch.mean((Ht@WOUT - (Hxx@WOUT) @ get_block_matrix(torch.tensor(1.)).t()) ** 2)

        print(torch.mean(loss_test**2))
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        pc=ax[0].contourf(x_evals, y_evals, (out_pred[:,0]**2 + out_pred[:,1]**2).reshape(len(x_evals), len(y_evals)).t())
        ax[0].set_title('predicted solution')
        fig.colorbar(pc, ax=ax[0])

        new_pred = out_pred[:,0].reshape(len(x_evals), len(y_evals)).t()
        new_pred1 = out_pred[:, 1].reshape(len(x_evals), len(y_evals)).t()

        ax[1].plot(new_pred[0,:],new_pred1[0,:])

        ax[2].plot(new_pred[-1, :], new_pred1[-1, :])
        plt.show()
    # import numpy as np
    # from matplotlib import pyplot as plt
    # from matplotlib.animation import FuncAnimation
    #
    # plt.style.use('seaborn-pastel')
    # dpi = 100
    #
    # # fig = plt.figure()
    # # ax = plt.axes()
    # # line, = plt.plot([], [])
    # fig, ax = plt.subplots(figsize=(5,5))
    #
    # def animate(i):
    #     print(i)
    #     ax.clear()
    #     # plt.cla()
    #
    #     with torch.no_grad():
    #         unew = func(torch.ones(100) *i/0.1, torch.linspace(-10., 10., 100))
    #         # a1 = plt.scatter(unew[:, 0], unew[:, 1])
    #
    #     ax.plot(unew[:,0])
    #     ax.plot(unew[:,1])
    #     # return figs
    #
    #
    # anim = FuncAnimation(fig, animate,frames=10,interval=100)
    #
    # anim.save('sine_wave.gif', writer='imagemagick')

    #
    # x_evals = torch.linspace(xl, xr, 50)
    # y_evals = torch.linspace(t0, tmax, 50)
    # x_evals.requires_grad = True
    # y_evals.requires_grad = True
    # grid_x, grid_t = torch.meshgrid(x_evals, y_evals)
    # # grid_x.requires_grad = True
    # # grid_t.requires_grad = True
    #
    # # print(grid_t[:,0], grid_x[:,0])
    #
    # grid_xx = grid_x.ravel()
    # grid_tt = grid_t.ravel()
    #
    # WOUT,Htt,Hxx = wout_gen.get_wout(func,grid_tt.reshape(-1,1),grid_xx.reshape(-1,1),grid_t,grid_x,)
    #
    # H = torch.cat([func.hidden_states(grid_tt.reshape(-1,1),grid_xx.reshape(-1,1)),torch.ones(len(Htt),1)],1)
    #
    # # Htt = diff(H,grid_tt.reshape(-1,1),2)
    # # Hxx = diff(H,grid_xx.reshape(-1,1),2)
    #
    # with torch.no_grad():
    #     out_pred = (H@WOUT)
    #     # print((out_pred))
    #     loss_test =Htt@WOUT -c(grid_xx.reshape(-1,1))*Hxx@WOUT
    #     print(torch.mean(loss_test**2))
    #
    #     fig,ax = plt.subplots(1,3,figsize=(15,5))
    #
    #     pc=ax[0].contourf(x_evals, y_evals, out_pred.reshape(len(x_evals), len(y_evals)).t())
    #     ax[0].set_title('predicted solution')
    #     fig.colorbar(pc, ax=ax[0])
    #
    #     u_true = 3*torch.sin(grid_xx) * torch.exp(-grid_tt)
    #
    #     ax[1].contourf(x_evals, y_evals, u_true.reshape(len(x_evals), len(y_evals)).t())
    #     ax[1].set_title('true solution')
    #
    #     pc = ax[2].contourf(x_evals, y_evals, (u_true.reshape(len(x_evals), len(y_evals)).t() - out_pred.reshape(len(x_evals), len(y_evals)).t()) ** 2)
    #     ax[2].set_title('residuals')
    #     fig.colorbar(pc, ax=ax[2])
    #
    #     plt.show()