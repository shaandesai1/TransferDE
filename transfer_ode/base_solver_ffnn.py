"""
base solver for transfer ode
"""
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import numpy as np
torch.manual_seed(33)

parser = argparse.ArgumentParser('NeuralODE transfer demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--tmax', type=float, default=2.)
parser.add_argument('--dt', type=int, default=0.1)

parser.add_argument('--method_rc', type=str, choices=['euler'], default='euler')
parser.add_argument('--wout', type=str, default='analytic')
parser.add_argument('--paramg', type=str, default='exp')

parser.add_argument('--niters', type=int, default=10000)
# parser.add_argument('--niters_test', type=int, default=15000)

parser.add_argument('--hidden_size', type=int, default=100)

parser.add_argument('--test_freq', type=int, default=100)

parser.add_argument('--viz', action='store_false')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_false')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint




import time
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
    def forward(self, t, y):
        # y = y[:, 0]
        yd = (-self.a0(t) * y + self.f(t)) / self.a1(t)
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
    function to learn the hidden states derivatives hdot
    """

    def __init__(self, number_dims):
        super(ODEFunc, self).__init__()
        self.number_dims = number_dims
        self.nl = nn.Tanh()
        self.lin1 = nn.Linear(1,number_dims)
        self.lin2 = nn.Linear(number_dims,number_dims)
        self.lout = nn.Linear(number_dims,1,bias=False)
    def forward(self, t):
        x = self.lin1(t)
        x = self.nl(x)
        x = self.lin2(x)
        x = self.nl(x)
        x = self.lout(x)
        return x
    def dot(self, t):
        outputs = self.forward(t)
        doutdt = [torch.autograd.grad(outputs[:,i].sum(),t,create_graph=True)[0] for i in range(outputs.shape[1])]
        return torch.cat(doutdt,1)

    def wouts(self, x):
        return self.lout(x)


    def h(self,t):
        x = self.lin1(t)
        x = self.nl(x)
        x = self.lin2(x)
        x = self.nl(x)
        return x

    def hdot(self,t):
        outputs = self.h(t)
        doutdt = [torch.autograd.grad(outputs[:, i].sum(), t, create_graph=True)[0] for i in range(outputs.shape[1])]
        return torch.cat(doutdt, 1)


class Transformer_Learned(nn.Module):
    """
    returns Wout learnable, only need hidden and output dims
    """

    def __init__(self, input_dims, output_dims):
        super(Transformer_Learned, self).__init__()
        self.lin1 = nn.Linear(args.hidden_size,output_dims)
    def forward(self,x):
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



    def get_wout(self, s, sd, y0, t):
        y0 = torch.stack([y0 for _ in range(len(s))]).reshape(len(s),-1)
        a1 = self.a1(t).reshape(-1, 1)
        a0 = self.a0(t).reshape(-1, 1)
        f = self.f(t).reshape(-1, 1)

        DH = (a1 * sd + a0 * s)
        D0 = (-f).repeat_interleave(y0.shape[1]).reshape(-1,y0.shape[1])
        lambda_0 = self.lambda_

        h0m = s[0].reshape(-1,1)
        W0 = torch.linalg.solve(DH.t() @ DH +lambda_0 +h0m@h0m.t(), -DH.t() @ D0 + h0m@(y0[0,:].reshape(1,-1))  )
        return W0


class Parametrization:

    def __init__(self, type='exp'):
        if type == 'exp':
            self.g = lambda t: (1. - torch.exp(-t))
            self.gd = lambda t: torch.exp(-t)
        elif type == 'lin':
            self.g = lambda t: t
            self.gd = lambda t: 1. + t*0

    def get_g(self, t):
        return self.g(t)

    def get_gdot(self, t):
        return self.gd(t)

    def get_g_gdot(self,t):
        return self.g(t),self.gd(t)

def compute_h_hdot(func,batch_t):

    integ = func.h(batch_t)
    integdot = func.hdot(batch_t)

    return integ,integdot

def compute_s_sdot(func,batch_t,param):

    integ = func(batch_t)
    integdot = func.dot(batch_t)

    return integ,integdot

if args.viz:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr,lst,s):
    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        for i in range(1):
            ax_traj.plot(t.detach().cpu().numpy(), true_y.cpu().numpy()[:, i, 0],
                         'g-')
            ax_traj.plot(t.detach().cpu().numpy(), pred_y.cpu().numpy()[:, i, 0], '--', 'b--')
        ax_phase.set_yscale('log')
        ax_phase.plot(np.arange(len(lst)),lst)

        ax_traj.legend()



        plt.draw()
        plt.pause(0.001)


if __name__ == '__main__':

    ii = 0
    NDIMZ = args.hidden_size
    # define coefficients as lambda functions, used for gt and wout_analytic
    a0 = lambda t: t**2#-(5./t + t)#-3*t**2
    a1 = lambda t:1. + 0.*t
    f = lambda t: torch.sin(t)#t**6#3*t**2#torch.sin(t)

    diffeq_init = diffeq(a0,a1,f)
    gt_generator = base_diffeq(diffeq_init)
    estim_generator = estim_diffeq(diffeq_init)
    true_y0 = torch.tensor([[5.]])
    t = torch.arange(0.,args.tmax,args.dt).reshape(-1,1)
    t.requires_grad = True
    true_y = gt_generator.get_solution(true_y0,t.ravel())
    wout_gen = Transformer_Analytic(a0, a1, f, 0.0)
    func = ODEFunc(NDIMZ)

    optimizer = optim.Adam(func.parameters(),lr=1e-3,weight_decay=1e-6)

    param = Parametrization(args.paramg)
    loss_collector = []
    for itr in range(1, args.niters + 1):
        func.train()
        t0 = torch.tensor([[0.]])
        t0.requires_grad = True
        tv = args.tmax*torch.rand(int(args.tmax/args.dt)).reshape(-1, 1)
        tv.requires_grad = True
        tv = torch.cat([t0,tv],0)

        optimizer.zero_grad()
        s,sd = compute_s_sdot(func,tv,param)
        pred_y = s
        pred_ydot = sd
        lst = (a1(tv.detach()).reshape(-1,1))*pred_ydot + (a0(tv.detach()).reshape(-1,1))*pred_y - f(tv.detach()).reshape(-1,1)
        loss = torch.mean(torch.square(lst)) + torch.mean(torch.square(pred_y[0,0]-true_y0[0,0]))
        loss.backward()
        optimizer.step()
        loss_collector.append(torch.square(lst).mean().item())
        if itr % args.test_freq == 0:
            func.eval()
            print(loss_collector[-1])
            s, sd = compute_s_sdot(func, t, param)
            pred_y = s.detach()
            pred_y = pred_y.reshape(-1,1,1)
            visualize(true_y.detach(), pred_y.detach(), func, ii,loss_collector,s)
            ii += 1



        # torch.save(func.state_dict(), 'func_dict_wout')

    # with torch.no_grad():

    a0 = lambda t: 3*t ** 4  # -(5./t + t)#-3*t**2
    a1 = lambda t: 1. + 0. * t
    f = lambda t: 0.*torch.sin(t)  # t**6#3*t**2#torch.sin(t)

    diffeq_init = diffeq(a0, a1, f)
    gt_generator = base_diffeq(diffeq_init)
    estim_generator = estim_diffeq(diffeq_init)
    t = torch.arange(0., args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True
    wout_gen = Transformer_Analytic(a0, a1, f, 0.0)

    func.eval()
    h, hd = compute_h_hdot(func, t)
    h=h.detach()
    hd=hd.detach()
    # print(h,hd)
    # print(h.shape,hd.shape)
    ics = torch.linspace(3.,7.,200).reshape(1,-1)

    # mdl = Transformer_Learned(1,ics.shape[1])
    # optimizer = optim.Adam(mdl.parameters(), lr=1e-3)

    #inference for other ICs
    loss_collector = []
    # s1 = time.time()
    # for itr in range(1, args.niters_test + 1):
    #     func.train()
    #     optimizer.zero_grad()
    #     pred_y = mdl(h)
    #     pred_ydot = mdl(hd)
    #     lst = (a1(t.detach()).reshape(-1, 1)) * pred_ydot + (a0(t.detach()).reshape(-1, 1)) * pred_y - f(
    #         t.detach()).reshape(-1, 1)
    #     loss = torch.mean(torch.square(lst)) + torch.mean(torch.square(pred_y[0, :] - ics[0, :]))
    #     loss.backward()
    #     optimizer.step()
    #     loss_collector.append(torch.square(lst).mean().item())
    #     print(loss.item())
    s2 = time.time()
    # results_df = np.zeros((len(ics),6))
    y0 = ics.reshape(1,-1)
    s1 = time.time()
    wout = wout_gen.get_wout(h, hd, y0, t.detach())
    pred_y = h@wout#torch.mm(h, wout)
    # print(pred_y.shape)
    s2 = time.time()
    print(f'all_ics:{s2-s1}')

    fig,ax = plt.subplots(1,3,figsize=(15,7))
    rmsr = 0.
    true_ys= torch.zeros(len(pred_y),len(ics))
    estim_ys = torch.zeros(len(pred_y), len(ics))
    # print(true_ys.shape)
    s1 = time.time()
    # y0 = ic.reshape(1,1)
    true_y = gt_generator.get_solution(ics.reshape(-1,1),t.ravel())
    true_ys = true_y.reshape(len(pred_y),ics.shape[1])
    s2 = time.time()
    print(f'gt_ics:{s2 - s1}')

    s1 = time.time()
    # for ic_index, ic in enumerate(ics):
    #     y0 = ic.reshape(1, 1)
    true_y = estim_generator.get_solution(ics.reshape(-1,1), t.ravel())
    estim_ys = true_y.reshape(len(pred_y),ics.shape[1])
    s2 = time.time()
    print(f'estim_ics:{s2 - s1}')

    print(f'prediction_accuracy:{((pred_y - true_ys) ** 2).mean()} pm {((pred_y - true_ys) ** 2).std()}')
    print(f'estim_accuracy:{((estim_ys - true_ys) ** 2).mean()} pm {((estim_ys - true_ys) ** 2).std()}')

