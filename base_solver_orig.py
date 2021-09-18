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
parser.add_argument('--tmax', type=float, default=5.)
parser.add_argument('--dt', type=int, default=0.1)

parser.add_argument('--method_rc', type=str, choices=['euler'], default='euler')
parser.add_argument('--wout', type=str, default='analytic')
parser.add_argument('--paramg', type=str, default='lin')

parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=200)

parser.add_argument('--test_freq', type=int, default=1)

parser.add_argument('--viz', action='store_false')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_false')
args = parser.parse_args()
import time
s0 = time.time()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint





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
            true_y = odeint(self.base, true_y0, t, method='euler')
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
        self.upper = nn.Linear(self.number_dims, self.number_dims, bias=True)
        self.upper1 = nn.Linear(self.number_dims, self.number_dims, bias=True)

        self.lower = nn.Sequential(nn.Linear(1, 1, bias=True))
        self.linear = nn.Linear(self.number_dims,self.number_dims,bias=False)
        self.nl = nn.Tanh()
        # self.drop = nn.Dropout(p=0.2)
    def forward(self, t, y):
        first = self.upper(y)
        second = self.lower(t.reshape(-1, 1))
        x= self.nl(first + second)
        x = self.upper1(x)
        x = self.nl(x)
        return x

    def get_z0(self, x):
        return self.linear(x)
        # first = self.upper(y)
        # second = self.lower(t.reshape(-1, 1))
        # return nn.Tanh()(first + second)


class Transformer_Learned(nn.Module):
    """
    returns Wout learnable, only need hidden and output dims
    """

    def __init__(self, hidden_dims, output_dims):
        super(Transformer_Learned, self).__init__()
        self.wout = nn.Parameter(torch.zeros(hidden_dims + 1, output_dims))
        self.wout = nn.init.kaiming_normal_(self.wout)

    def get_wout(self):
        return self.wout


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
        D0 = (a0 * y0 - f)
        lambda_0 = self.lambda_
        W0 = torch.linalg.solve(DH.t() @ DH + lambda_0, -DH.t() @ D0)
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


def compute_s_sdot(func,zinit,batch_t,param):
    # batch_t = torch.arange(0., 2*args.tmax, args.dt).reshape(-1, 1)

    integ = odeint(func, zinit, batch_t.ravel(), method=args.method_rc)
    # integ[integ<0.1] = 0
    integdot = torch.stack([func(batch_t.ravel()[i], integ[i]) for i in range(len(integ))])

    integ = integ.reshape(-1, NDIMZ)
    integdot = integdot.reshape(-1, NDIMZ)

    bias = torch.ones((len(batch_t), 1))
    integ = torch.cat([integ, bias], 1)

    bias2 = torch.zeros((len(batch_t), 1))
    integdot = torch.cat([integdot, bias2], 1)

    gt = param.get_g(t).repeat_interleave(NDIMZ+1 ).reshape(-1, NDIMZ+1 )
    gtd = param.get_gdot(t).repeat_interleave(NDIMZ+1 ).reshape(-1, NDIMZ+1 )

    s = gt * integ
    sd = gtd * integ + gt * integdot

    return s,sd
    # return integ,integdot

if args.viz:
    # makedirs('png')
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
            ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, i, 0],
                         'g-')
            ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, i, 0], '--', 'b--')
        # print(lst)
        ax_phase.set_yscale('log')

        ax_phase.plot(np.arange(len(lst)),lst)

        # ax_vecfield.hist(s.cpu().numpy())
        # ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        # ax_traj.set_ylim(-2, 2)
        ax_traj.legend()



        plt.draw()
        plt.pause(0.001)


if __name__ == '__main__':

    ii = 0
    NDIMZ = args.hidden_size
    # define coefficients as lambda functions, used for gt and wout_analytic
    a0 = lambda t: t**2#-(5./t + t)#-3*t**2
    a1 = lambda t:1 + 0.*t
    f = lambda t: torch.sin(t)#t**6#3*t**2#torch.sin(t)

    diffeq_init = diffeq(a0,a1,f)
    gt_generator = base_diffeq(diffeq_init)
    estim_generator = estim_diffeq(diffeq_init)
    true_y0 = torch.tensor([[5.]])
    t = torch.arange(0.,args.tmax,args.dt).reshape(-1,1)

    true_y = gt_generator.get_solution(true_y0,t.ravel())

    # wout generator
    if args.wout == 'analytic':
        wout_gen = Transformer_Analytic(a0, a1, f, 0.0)
    else:
        wout_gen = Transformer_Learned(NDIMZ, true_y0.shape[1])
    # hidden state generator
    func = ODEFunc(NDIMZ)
    # func.upper.weight

    if args.wout  == 'analytic':
        optimizer = optim.Adam(func.parameters(),lr=1e-5)
    # elif args.wout == 'learned':
    #     optimizer = optim.SGD([
    #         {'params': func.parameters()},
    #         {'params': wout_gen.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3}
    #     ], lr=1e-3)


    # true_y0 = torch.tensor([[3.]])
    param = Parametrization(args.paramg)
    zinit_ = torch.ones(NDIMZ).reshape(1, NDIMZ)
    loss_collector = []
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()

        zinit = func.get_z0(zinit_)

        s,sd = compute_s_sdot(func,zinit,t,param)
        # print(s[0],sd[0])
        # if (itr-1)%200==0:
        wout = wout_gen.get_wout(s.detach(),sd.detach(), true_y0, t)

        # wout = torch.ones(100,1)
        # wout = wout_gen.get_wout(s.detach(), sd.detach(), true_y0, t)
        # print(wout)
        # wout = wout.detach()

        pred_y = true_y0 + torch.mm(s, wout)
        pred_ydot = torch.mm(sd, wout)
        lst = (a1(t).reshape(-1,1))*pred_ydot + (a0(t).reshape(-1,1))*pred_y - f(t).reshape(-1,1)
        # lst = wout-1.
        # l2_reg = None
        # l2_reg = func.upper.weight.norm(1)

        # for name,paramss in func.named_parameters():
        #     if name == 'upper.weight':
        #         # print(paramss)
        #         l2_reg = torch.sum(torch.abs(paramss))
        # print(func.upper.weight.detach().cpu().numpy())
        # print(l2_reg)
        loss = torch.mean(torch.square(lst)) #+ .001*l2_reg
        loss.backward()
        optimizer.step()
        loss_collector.append(torch.square(lst).mean().item())
        if itr % args.test_freq == 0:
            # print(loss.item())
            with torch.no_grad():
                s, sd = compute_s_sdot(func, zinit, t, param)

                wout = wout_gen.get_wout(s, sd, true_y0, t)
                #     # wout = torch.ones(100, 1)
                #
                # elif args.wout == 'learned':
                #     wout = wout_gen.get_wout()

                pred_y = true_y0[0,0].reshape(1, 1) + torch.mm(s, wout)[:,0]
                pred_y = pred_y.reshape(-1,1,1)
                # loss = torch.mean(torch.abs(pred_y - true_y))
                # print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii,loss_collector,s)
                ii += 1



        # torch.save(func.state_dict(), 'func_dict_wout')

    with torch.no_grad():

        s, sd = compute_s_sdot(func, zinit, t, param)
        #inference for other ICs
        ics = torch.linspace(3.,7.,200)


        # results_df = np.zeros((len(ics),6))
        y0 = ics.reshape(1,-1)
        s1 = time.time()
        wout = wout_gen.get_wout(s, sd, y0, t)
        pred_y = y0 + torch.mm(s, wout)
        s2 = time.time()
        print(f'all_ics:{s2-s1}')

        fig,ax = plt.subplots(1,3,figsize=(15,7))
        rmsr = 0.
        true_ys= torch.zeros(len(pred_y),len(ics))
        estim_ys = torch.zeros(len(pred_y), len(ics))
        # print(true_ys.shape)
        s1 = time.time()
        # y0 = ic.reshape(1,1)
        true_y = gt_generator.get_solution(y0.reshape(-1,1),t.ravel())
        true_ys = true_y.reshape(len(pred_y),len(ics))
        s2 = time.time()
        print(f'gt_ics:{s2 - s1}')

        s1 = time.time()
        # for ic_index, ic in enumerate(ics):
        #     y0 = ic.reshape(1, 1)
        true_y = estim_generator.get_solution(y0.reshape(-1,1), t.ravel())
        estim_ys = true_y.reshape(len(pred_y),len(ics))
        s2 = time.time()
        print(f'estim_ics:{s2 - s1}')

        print(f'prediction_accuracy:{((pred_y - true_ys) ** 2).mean()} pm {((pred_y - true_ys) ** 2).std()}')
        print(f'estim_accuracy:{((estim_ys - true_ys) ** 2).mean()} pm {((estim_ys - true_ys) ** 2).std()}')
        print(f'total time {s1 - s0}')

        # s1 = time.time()
            # if args.wout == 'analytic':
            #     wout = wout_gen.get_wout(s, sd, y0, t)
            # elif args.wout == 'learned':
            #     wout = wout_gen.get_wout()
            # pred_y = y0.reshape(1, 1) + torch.mm(s, wout)
            # pred_ydot = torch.mm(sd,wout)
            # pred_y =
            # s2 = time.time()
            # print(f'pred time: {s2-s1}')
            # results_df[ic_index, 2] = s2 - s1



            # results_df[ic_index, 3] = (((pred_y[:,ic_index] - true_y) ** 2).mean()).cpu().numpy()
            # results_df[ic_index,4] = (((estim_y - true_y) ** 2).mean()).cpu().numpy()
            # results_df[ic_index, 5] =rmsr_ic.mean().cpu().numpy()

    #         ax[0].plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0],
    #                      'g-')
    #         ax[0].plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', 'b--')
    #         ax[0].set_ylabel('y(t)')
    #         ax[0].set_xlabel('t')
    #         residual = ((true_y - pred_y)**2).cpu().numpy()
    #         ax[1].plot(t.cpu().numpy(),residual.reshape(-1,1),'--')
    #         ax[1].set_yscale('log')
    #         ax[1].set_xlabel('time')
    #         ax[1].set_ylabel('MSE Error')
    #
    #     ax[2].plot(t.cpu().numpy(), np.sqrt(rmsr.reshape(-1, 1)/len(ics)), '--')
    #     ax[2].set_yscale('log')
    #     ax[2].set_xlabel('time')
    #     ax[2].set_ylabel('RMSR')
    #     plt.show()
    #
    # np.save('results_df_base.npy',results_df)