"""
base solver for transfer ode (first order methods)
"""
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import numpy as np
import time
import seaborn as sns
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser('transfer demo')

parser.add_argument('--tmax', type=float, default=3.)
parser.add_argument('--dt', type=int, default=0.1)
parser.add_argument('--niters', type=int, default=40000)
parser.add_argument('--niters_test', type=int, default=15000)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--num_ics', type=int, default=3)
parser.add_argument('--num_test_ics', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--bs', type=int, default=100)

parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--evaluate_only', action='store_false')

args = parser.parse_args()

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# torch.backends.cudnn.benchmark = True


# device =



torch.set_default_tensor_type('torch.DoubleTensor')


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
        u = self.nl2(u)#self.weight_1*self.nl1(u) + (1.-self.weight_1)*self.nl2(u)
        u = self.lin2(u)
        u = self.weight_2*self.nl1(u) + (1.-self.weight_2)*self.nl2(u)
        return u

    def forward(self, t,x):
        u = self.hidden_states(t,x)
        u = self.lout(u)
        return u

    def wouts(self, x):
        return self.lout(x)


def psi(t,x,m,sig,p0,x0=0.):
    E = p0**2/2/m
    fac1= np.pi**(-1/4)/np.sqrt(sig*(1+1j*t/(m*sig**2)))
    fac2= - (x-(x0+p0*t/m))**2/(2*sig**2*(1+1j*t/(m*sig**2)))
    fac3= 1j*(p0*x-E*t)
    return fac1*np.exp(fac2)*np.exp(fac3)




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

        # t,x = grid_t[:, -1].reshape(-1, 1), grid_x[:, -1].reshape(-1, 1)

        # t,x = torch.linspace(0,1,100,requires_grad=True).reshape(-1,1),torch.linspace(-10,10,100,requires_grad=True).reshape(-1,1)
        zindices = np.random.choice(len(t),350,replace=False)
        # print(zindices)
        t = t[zindices,:].reshape(-1,1)
        x = x[zindices,:].reshape(-1,1)

        # H = func.hidden_states(t, x)  # torch.cat([func.hidden_states(t,x),torch.ones(len(t),1)],1)
        # dHdt = diff(H,t)#torch.cat([diff(H,t,1),torch.zeros(len(t),1)],1)
        # d2Hdx2 =diff(H,x,2)#torch.cat([diff(H,x,2),torch.zeros(len(t),1)],1)
        #
        # # H = torch.cat([H,torch.ones(len(t),1)],1)
        #
        # Amatrix = get_block_matrix(torch.tensor(1.))
        #
        # print(Amatrix)
        #
        # HHt = torch.block_diag(dHdt, dHdt)
        # HHxx = torch.block_diag(d2Hdx2, d2Hdx2)
        # HH = torch.block_diag(H,H)
        #
        # # print(HHt.shape,HHxx.shape,HH.shape)
        #
        #
        # Amatrixhat = torch.zeros((HH.shape[0], HH.shape[0]))
        # # print(Amatrix.shape)
        # for i in range(Amatrix.shape[0]):
        #     for j in range(Amatrix.shape[1]):
        #         # print(Amatrix[i,j])
        #         Amatrixhat[i * H.shape[0]:(i + 1) * H.shape[0], j * H.shape[0]:(j + 1) * H.shape[0]] = torch.eye(
        #             H.shape[0], H.shape[0]) * Amatrix[i, j]
        #
        # # DH = hddothat + Amatrixhat @ hhat
        # # print(Amatrix)
        # # print(Amatrixhat)
        # DH = (HHt-Amatrixhat@HHxx)
        #
        # print(grid_t[:,0],grid_x[:,0])
        # H0 = func.hidden_states(grid_t[:, 0].reshape(-1, 1), grid_x[:, 0].reshape(-1, 1))
        # # H0 = self.append_ones(H0)
        # HH0 = torch.block_diag(H0,H0)
        #
        # lbc = grid_x[0, :].reshape(-1, 1)
        # HL = func.hidden_states(grid_t[0, :].reshape(-1, 1),lbc)
        # HLd = diff(HL, lbc)
        #
        # # HL = self.append_ones(HL)
        # HHL = torch.block_diag(HL, HL)
        # # HLd = self.append_ones(HLd,'zeros')
        # HHLd = torch.block_diag(HLd, HLd)
        # # print(HLd)
        #
        # rbc = grid_x[-1, :].reshape(-1, 1)
        # # print(rbc)
        # HR = func.hidden_states(grid_t[-1, :].reshape(-1, 1), rbc)
        # HRd = diff(HR, rbc)
        # # print(HRd)
        #
        # # HR = self.append_ones(HR)
        # HHR = torch.block_diag(HR, HR)
        # # HRd = self.append_ones(HRd,'zeros')
        # # print(HRd[:,-1])
        # HHRd = torch.block_diag(HRd, HRd)
        #
        # full_IC_matrix = []
        # for sigma_ in sigma:
        #     for p0_ in p0:
        #         print(sigma_)
        #         IC = get_ic(grid_t[:, 0].reshape(-1, 1), grid_x[:, 0].reshape(-1, 1), sigma=sigma_, x0=0., p0=p0_)
        #         new_IC = torch.cat([IC[:,0].ravel(),IC[:,1].ravel()]).reshape(-1,1)
        #
        #         print(IC[:,0])
        #         print(new_IC)
        #         full_IC_matrix.append(new_IC)
        #
        # full_IC_matrix = torch.hstack(full_IC_matrix)
        #
        # print('full ic matrix')
        # print(full_IC_matrix.shape)
        #
        # LVEC = DH.t() @ DH + HH0.t() @ HH0 + (HHL-HHR).t()@(HHL-HHR) + (HHLd-HHRd).t()@(HHLd-HHRd)
        # #
        # # print(torch.linalg.cond(LVEC))
        # #
        # W0 = torch.linalg.solve(LVEC,HH0.t()@full_IC_matrix)
        # #
        # # # nwout = torch.cat([W0[:args.hidden_size + 1, :].reshape(-1, ), W0[args.hidden_size + 1:, 0].reshape(-1, 1)], 1)
        # #
        # return W0,dHdt,d2Hdx2

        H = func.hidden_states(t,x)#torch.cat([func.hidden_states(t,x),torch.ones(len(t),1)],1)
        dHdt = torch.cat([diff(H,t,1),torch.zeros(len(t),1)],1)
        d2Hdx2 =torch.cat([diff(H,x,2),torch.zeros(len(t),1)],1)

        H = torch.cat([H,torch.ones(len(t),1)],1)

        Amatrix = get_block_matrix(torch.tensor(1.))

        # print(Amatrix)

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
        # print(Amatrix)
        # print(Amatrixhat)
        DH = (HHt-Amatrixhat@HHxx)

        # print(grid_t[:,0],grid_x[:,0])
        H0 = func.hidden_states(grid_t[:, 0].reshape(-1, 1), grid_x[:, 0].reshape(-1, 1))
        H0 = self.append_ones(H0)
        HH0 = torch.block_diag(H0,H0)

        lbc = grid_x[0, :].reshape(-1, 1)
        HL = func.hidden_states(grid_t[0, :].reshape(-1, 1),lbc)
        HLd = diff(HL, lbc)

        HL = self.append_ones(HL)
        HHL = torch.block_diag(HL, HL)
        HLd = self.append_ones(HLd,'zeros')
        HHLd = torch.block_diag(HLd, HLd)
        # print(HLd)

        rbc = grid_x[-1, :].reshape(-1, 1)
        # print(rbc)
        HR = func.hidden_states(grid_t[-1, :].reshape(-1, 1), rbc)
        HRd = diff(HR, rbc)
        # print(HRd)

        HR = self.append_ones(HR)
        HHR = torch.block_diag(HR, HR)
        HRd = self.append_ones(HRd,'zeros')
        # print(HRd[:,-1])
        HHRd = torch.block_diag(HRd, HRd)

        full_IC_matrix = []
        for sigma_ in sigma:
            for p0_ in p0:
                # print(sigma_)
                IC = get_ic(grid_t[:, 0].reshape(-1, 1), grid_x[:, 0].reshape(-1, 1), sigma=sigma_, x0=0., p0=p0_)
                new_IC = torch.cat([IC[:,0].ravel(),IC[:,1].ravel()]).reshape(-1,1)

                # print(IC[:,0])
                # print(new_IC)
                full_IC_matrix.append(new_IC)

        full_IC_matrix = torch.hstack(full_IC_matrix)

        # print('full ic matrix')
        # print(full_IC_matrix.shape)

        # print(DH.shape,HH0.shape,HHL.shape,HHR.shape,HHLd.shape,HHRd.shape,new_IC.shape)
        #
        # print(DH.t()@DH,HH0.t()@HH0)


        LVEC = DH.t() @ DH + HH0.t() @ HH0 + (HHL-HHR).t()@(HHL-HHR) + (HHLd-HHRd).t()@(HHLd-HHRd)

        print(torch.linalg.cond(LVEC))

        W0 = torch.linalg.solve(LVEC,HH0.t()@full_IC_matrix)

        # nwout = torch.cat([W0[:args.hidden_size + 1, :].reshape(-1, ), W0[args.hidden_size + 1:, 0].reshape(-1, 1)], 1)

        return W0,dHdt,d2Hdx2


        # sp1 = full_IC_matrix.shape[1]
        # new_mat_A = torch.cat([DH,HH0,HHL-HHR,HHLd-HHRd],0)
        # new_mat_Y = torch.cat([torch.zeros(len(DH),sp1),full_IC_matrix,torch.zeros(len(HHL),sp1),torch.zeros(len(HHL),sp1)],0)
        # W0 = torch.linalg.lstsq(new_mat_A,new_mat_Y)#torch.linalg.solve(LHS, DH.t()@rho + H0.t()@BB + HT.t()@TB + HL.t()@BL + HR.t()@BR)
        # return W0.solution, dHdt, d2Hdx2






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
        # ax_phase.contourf(x, t, (u[:, 0]**2 + u[:,1]**2).reshape(len(x), len(t)).t())

        # ps1= get_ic(torch.ones(500)*0,torch.linspace(-10.,10.,500),0.5,x0=0.,p0=1.)
        ps11 = psi(torch.ones(500)*0,torch.linspace(-10.,10.,500), 1, 0.5, 1, x0=0)
        ps2 = psi(torch.ones(500) * 0, torch.linspace(-10., 10., 500), 1, 0.6, 2, x0=0)
        ps3 = psi(torch.ones(500) * 0, torch.linspace(-10., 10., 500), 1, 0.7, 3, x0=0)

        unew = func(torch.ones(500)*0,torch.linspace(-10.,10.,500))

        with torch.no_grad():
        # unew1 = func(torch.ones(50) * 2., torch.linspace(-10., 10., 50))
            ax_traj.plot(unew[:, 0])
            ax_traj.plot(unew[:,1])
            ax_phase.plot(unew[:,2])
            ax_phase.plot(unew[:,3])
            # ax_vecfield.plot(unew[:,4],unew[:,5])

            # ax_traj.plot((ps1[:,0]),label='ti eqn')
            ax_traj.plot(np.real(ps11),label='td eqn')
            ax_traj.plot(np.imag(ps11), label='td eqn')
            ax_phase.plot(np.real(ps2))
            ax_phase.plot(np.imag(ps2))

            ps1 = psi(torch.ones(500) * .5, torch.linspace(-10., 10., 500), 1, 0.5, 1, x0=0)

            unew = func(torch.ones(500) * 0.5, torch.linspace(-10., 10., 500))

            ax_vecfield.plot(np.real(ps1))
            ax_vecfield.plot(np.imag(ps1))

            ax_vecfield.plot(unew[:,0])
            ax_vecfield.plot(unew[:,1])

        # ax_vecfield2.scatter(unew[:,0],unew[:,1])

        # ax_vecfield2.scatter(unew1[:, 0], unew1[:, 1])

        # ax_vecfield.contourf(x, t, (u_true.reshape(len(x), len(t)).t()-u.reshape(len(x),len(t)).t())**2)
        plt.legend()
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

    optimizer = optim.Adam(func.parameters(), lr=1e-3)

    loss_collector = []
    best_residual = 1e-1
    if not args.evaluate_only:

        init_force = torch.cat([get_ic(ic_t0, x_evals, sigma=0.5, p0=1.), get_ic(ic_t0, x_evals, sigma=0.6, p0=2.),
                                get_ic(ic_t0, x_evals, sigma=0.7, p0=3.), ], 1)

        for itr in range(1, args.niters + 1):
            func.train()
            indices = torch.tensor(np.random.choice(len(grid_x),1000))
            x_tr = (grid_x[indices]).reshape(-1,1) + 0.1*torch.rand(len(indices),1)
            t_tr = (grid_t[indices]).reshape(-1,1) + 0.005*torch.rand(len(indices),1)

            # add t0 to training times, including randomly generated ts
            optimizer.zero_grad()

            u = func(t_tr,x_tr)
            dudt = diff(u,t_tr,1)
            d2udx2 = diff(u, x_tr, 2)
            loss_diffeq = torch.mean((dudt - d2udx2@DIFF_MATRIX.t())**2)

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

    func.load_state_dict(torch.load('func_ffnn_schroed2',map_location=torch.device('cpu')))
    func.eval()

    tmax = 0.5

    x_evals = torch.linspace(xl, xr, 200)
    y_evals = torch.linspace(t0, tmax, 200)
    x_evals.requires_grad = True
    y_evals.requires_grad = True
    grid_x, grid_t = torch.meshgrid(x_evals, y_evals)
    # grid_x.requires_grad = True
    # grid_t.requires_grad = True


    sigmas =torch.tensor([0.5,0.6,0.7])#torch.tensor([0.5,0.6,0.7])#torch.linspace(0.5,0.9,100)
    p0s =torch.tensor([1.,2.,3.])#torch.tensor([1.,2.,3.])#torch.linspace(1.,4.,100)



    # print(grid_t[:,0], grid_x[:,0])
    wout_gen = Transformer_Analytic()
    grid_xx = grid_x.ravel()
    grid_tt = grid_t.ravel()

    # func, t, x, grid_t, grid_x, sigma, p0
    s1 = time.time()
    WOUT,Ht,Hxx = wout_gen.get_wout(func,grid_tt.reshape(-1,1),grid_xx.reshape(-1,1),grid_t,grid_x,sigmas,p0s)
    print(f'time:{time.time()-s1}')

    ### first figure

    # H = torch.cat([func.hidden_states(grid_tt.reshape(-1,1),grid_xx.reshape(-1,1)),torch.ones(len(Ht),1)],1)
    H = torch.cat([func.hidden_states(grid_tt.reshape(-1,1),grid_xx.reshape(-1,1)),torch.ones(len(grid_xx),1)],1)

    # print(WOUT.shape,H.shape,Ht.shape,Hxx.shape)
    HH = torch.block_diag(H,H)

    teval_point = .5

    error_vec = []

    import matplotlib

    matplotlib.rcParams['text.usetex'] = True

    sns.axes_style(style='ticks')
    sns.set_context("paper", font_scale=1.3,
                    rc={"font.size": 20, "axes.titlesize": 25, "axes.labelsize": 20, "axes.legendsize": 20,
                        'lines.linewidth': 3.})
    sns.set_palette('deep')
    sns.set_color_codes(palette='deep')


    fig,ax = plt.subplots(3,3,sharex=True,sharey=True)



    axs = ax.ravel()
    with torch.no_grad():
        out_pred = HH@WOUT
        # print(out_pred.shape)
        idx = 0
        for sigma_ in sigmas:
            for p0_ in p0s:
                # print(sigma_)
                gt_psi = psi(teval_point,x_evals, 1., sigma_, p0_, x0=0)
                gt_real = np.real(gt_psi)
                gt_img = np.imag(gt_psi)
                pred_psi = (out_pred[:,idx]).reshape(-1,1)
                pred_psi_real = (pred_psi[:len(grid_xx), 0]).reshape(len(x_evals), len(y_evals)).t()
                pred_psi_img = (pred_psi[len(grid_xx):, 0]).reshape(len(x_evals), len(y_evals)).t()


                # error = ((gt_real.ravel()-pred_psi_real[0,:].ravel()/norm_const)**2 + (gt_img.ravel()-pred_psi_img[0,:].ravel()/norm_const)**2).mean()
                # error_vec.append(error)

                # eval_func = func(1*torch.ones_like(x_evals).reshape(-1,1), x_evals.reshape(-1,1))

                if ([sigma_,p0_] == [0.5,1.]) or([sigma_,p0_] == [0.6,2.]) or ([sigma_,p0_] == [0.7,3.]):
                    axs[idx].plot(gt_real, gt_img, label='gt', c='black', linestyle='--')
                    axs[idx].plot(pred_psi_real[-1,:],pred_psi_img[-1,:],label='pred',color='g',alpha=0.9)

                # elif idx == 4:
                #     axs[idx].plot(gt_real,gt_img,label='gt',c='black',linestyle='--')
                #     axs[idx].plot(pred_psi_real[-1,:],pred_psi_img[-1,:],label='pred',color='green')
                # elif idx == 8:
                #     axs[idx].plot(gt_real,gt_img,label='gt',c='black',linestyle='--')
                #     axs[idx].plot(pred_psi_real[-1,:],pred_psi_img[-1,:],label='pred',color='green')
                else:
                    # print(norm_const)
                    axs[idx].plot(gt_real, gt_img, label='gt', c='black', linestyle='--')
                    axs[idx].plot(pred_psi_real[-1, :], pred_psi_img[-1, :], label='pred', color='b',alpha=0.9)


                # axs[idx].plot(pred_psi_real[-1,:],pred_psi_img[-1,:],label='pred')

                # axs[idx].plot(pred_psi_real[-1, :], pred_psi_img[-1, :], label='pred1')
                # axs[idx].plot(pred_psi_real[:, 0], pred_psi_img[:, 0], label='pred2')
                # axs[idx].plot(pred_psi_real[:, -1], pred_psi_img[:, -1], label='pred3')

                # axs[idx].plot(pred_psi_real[:, -1], pred_psi_img[:, -1], label='pred1')
                # axs[idx].set_xlabel(r'$\psi_R$')
                # axs[idx].set_ylabel(r'$\psi_I$')
                idx += 1
                # plt.legend()


        ax[0,0].set_ylabel(r'$\sigma=0.5$')
        ax[1, 0].set_ylabel(r'$\sigma=0.6$')
        ax[2, 0].set_ylabel(r'$\sigma=0.7$')

        ax[2, 0].set_xlabel(r'$p_0=1$')
        ax[2, 1].set_xlabel(r'$p_0=2$')
        ax[2, 2].set_xlabel(r'$p_0=3$')

        fig.subplots_adjust(wspace=0, hspace=0)
        # plt.tight_layout()
        plt.savefig('schroedinger.pdf',dpi=2400,bbox_inches='tight')

    # tmax = 1.
    #
    # x_evals = torch.linspace(xl, xr, 200)
    # y_evals = torch.linspace(t0, tmax, 200)
    # x_evals.requires_grad = True
    # y_evals.requires_grad = True
    # grid_x, grid_t = torch.meshgrid(x_evals, y_evals)
    #
    # WOUT, Ht, Hxx = wout_gen.get_wout(func, grid_t.reshape(-1, 1), grid_x.reshape(-1, 1), grid_t, grid_x, [0.65],[2.5])
    #
    # # H = torch.cat([func.hidden_states(grid_tt.reshape(-1,1),grid_xx.reshape(-1,1)),torch.ones(len(Ht),1)],1)
    # H = torch.cat([func.hidden_states(grid_t.reshape(-1, 1), grid_x.reshape(-1, 1)), torch.ones(len(grid_x.ravel()), 1)],1)
    #
    # # print(WOUT.shape,H.shape,Ht.shape,Hxx.shape)
    # HH = torch.block_diag(H, H)
    # with torch.no_grad():
    #     out_pred = HH @ WOUT
    #
    #     fig, ax = plt.subplots(1,2,figsize=(10,5))
    #
    #     # gt_psi = psi(teval_point, x_evals, 1., sigma_, p0_, x0=0)
    #     # gt_real = np.real(gt_psi)
    #     # gt_img = np.imag(gt_psi)
    #     pred_psi = (out_pred[:, 0]).reshape(-1, 1)
    #     pred_psi_real = (pred_psi[:len(grid_x.ravel()), 0]).reshape(len(x_evals), len(y_evals)).t()
    #     pred_psi_img = (pred_psi[len(grid_x.ravel()):, 0]).reshape(len(x_evals), len(y_evals)).t()
    #
    #     predwf = pred_psi_real ** 2 + pred_psi_img ** 2
    #     gtwf = (np.abs(psi(grid_t,grid_x,1.,0.65,2.5)))**2
    #     a1= ax[0].contourf(x_evals,y_evals,predwf)
    #     plt.colorbar(a1,ax=ax[0])
    #
    #     a2=ax[1].contourf(x_evals,y_evals,(np.transpose(gtwf) - predwf)**2)
    #     plt.colorbar(a2,ax=ax[1])
    #     plt.tight_layout()
    #     plt.savefig('residuals.pdf',dpi=2400,bbox_inches='tight')

    tmax = 1.

    sns.set_context("paper", font_scale=2.3,
                    rc={"font.size": 30, "axes.titlesize": 25, "axes.labelsize": 30, "axes.legendsize": 20,
                        'lines.linewidth': 3.})

    sigmas = torch.linspace(0.3,1.5,50)#torch.tensor([0.5,0.6,0.7])#torch.linspace(0.5,0.9,100)
    p0s = torch.linspace(.7,5.,50)#torch.tensor([1.,2.,3.])#torch.linspace(1.,4.,100)
    # sigmas = torch.linspace(0.2,1.5,50)#torch.tensor([0.5,0.6,0.7])#torch.linspace(0.5,0.9,100)
    # p0s = torch.linspace(.5,5,50)#torch.tensor([1.,2.,3.])#torch.linspace(1.,4.,100)

    x_evals = torch.linspace(xl, xr, 200)
    y_evals = torch.linspace(t0, tmax, 200)
    x_evals.requires_grad = True
    y_evals.requires_grad = True
    grid_x, grid_t = torch.meshgrid(x_evals, y_evals)

    # s1 = time.time()
    WOUT, Ht, Hxx = wout_gen.get_wout(func, grid_t.reshape(-1, 1), grid_x.reshape(-1, 1), grid_t, grid_x, sigmas,p0s)
    # print(f'time:{time.time()-s1}')
    #
    # # H = torch.cat([func.hidden_states(grid_tt.reshape(-1,1),grid_xx.reshape(-1,1)),torch.ones(len(Ht),1)],1)
    H = torch.cat([func.hidden_states(grid_t.reshape(-1, 1), grid_x.reshape(-1, 1)), torch.ones(len(grid_x.ravel()), 1)],1)
    #
    # # print(WOUT.shape,H.shape,Ht.shape,Hxx.shape)
    HH = torch.block_diag(H, H)
    from matplotlib import ticker, cm
    idx = 0
    with torch.no_grad():
        out_pred = HH @ WOUT

        # fig, ax = plt.subplots(1,2,figsize=(10,5))
        error = np.zeros(int(len(sigmas)*len(p0s)))
        errormeans = np.zeros(int(len(sigmas) * len(p0s)))

        for sigma_ in sigmas:
            for p0_ in p0s:
                pred_psi = (out_pred[:, idx]).reshape(-1, 1)
                pred_psi_real = (pred_psi[:len(grid_x.ravel()), 0]).reshape(len(x_evals), len(y_evals)).t()
                pred_psi_img = (pred_psi[len(grid_x.ravel()):, 0]).reshape(len(x_evals), len(y_evals)).t()

                predwf = pred_psi_real ** 2 + pred_psi_img ** 2
                gtwf = (np.abs(psi(grid_t,grid_x,1.,sigma_,p0_)))**2
                error[idx] =torch.max((np.transpose(gtwf) - predwf)**2)
                errormeans[idx] = torch.mean((np.transpose(gtwf) - predwf)**2)
                idx+=1
                # plt.colorbar(a2,ax=ax[1])
                # plt.tight_layout()
                # plt.savefig('residuals.pdf',dpi=2400,bbox_inches='tight')
    print('error')
    print(np.mean(errormeans),np.std(errormeans))

    # import matplotlib.pyplot as plt

    # sns.axes_style(style='ticks')
    # sns.set_context("paper", font_scale=3,
    #                 rc={"font.size": 30, "axes.titlesize": 25, "axes.labelsize": 30, "axes.legendsize": 20,
    #                     'lines.linewidth': 2.5})
    # sns.set_palette('deep')
    # sns.set_color_codes(palette='deep')



    fig,ax = plt.subplots(figsize=(8,6))
    # cs =ax[0].contourf(sigmas,p0s,np.transpose(error.reshape(len(sigmas),len(p0s))),locator=ticker.LogLocator())
    # cbar = fig.colorbar(cs,ax=ax[0])

    cs = ax.contourf(sigmas, p0s, np.transpose(errormeans.reshape(len(sigmas), len(p0s))),
                        locator=ticker.LogLocator())
    cbar = fig.colorbar(cs, ax=ax)
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'$p_0$')
    ax.scatter([.5,.6,.7], [1,2,3],s=100, c='r')
    plt.savefig('schrd_abl.pdf',dpi=2400,bbox_inches='tight')
