"""
base solver for transfer ode
"""
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(33)

parser = argparse.ArgumentParser('NeuralODE transfer demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='rk4')
parser.add_argument('--tmax', type=float, default=3.*1.)
parser.add_argument('--dt', type=int, default=0.005)

parser.add_argument('--method_rc', type=str, choices=['euler','rk4'], default='euler')
parser.add_argument('--wout', type=str, default='learned')
parser.add_argument('--paramg', type=str, default='exp')

parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--niters_wout', type=int, default=15000)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--test_freq', type=int, default=50)

parser.add_argument('--viz', action='store_true')
parser.add_argument('--evaluate_only', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_false')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint



device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

print(device)

SCALER = 1.

class diffeq(nn.Module):
    """
    defines the diffeq of interest
    """

    def __init__(self):
        super().__init__()

    # ((pred_qd - pred_p) ** 2) + ((pred_p + pred_q + pred_q ** 3) ** 2)
    # return ydot
    def forward(self, t, y):
        q,p = y[:, 0],y[:,1]
        qd = p
        pd = -q - SCALER*q**3
        return torch.cat([qd.reshape(-1,1),pd.reshape(-1,1)],1)


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
    function to learn the hidden states derivatives hdot
    """

    def __init__(self, number_dims):
        super(ODEFunc, self).__init__()
        # self.alpha = 0.1
        self.number_dims = number_dims
        self.upper = nn.Sequential(
            nn.Linear(self.number_dims+1, 2*self.number_dims,bias = True),
            SiLU(),
            nn.Linear(self.number_dims*2, self.number_dims,bias = True))

    def forward(self, t, y):
        first = self.upper(torch.cat([y,t.reshape(-1,1)],1))
        return SiLU()(first)


class Transformer_Learned(nn.Module):
    """
    returns Wout learnable, only need hidden and output dims
    """

    def __init__(self, hidden_dims, output_dims,num_systems):
        super(Transformer_Learned, self).__init__()
        self.wout = nn.Parameter(torch.zeros(hidden_dims + 1, output_dims))
        self.wout = nn.init.kaiming_normal_(self.wout)

        self.wout1 = nn.Parameter(torch.zeros(hidden_dims + 1, output_dims))
        self.wout1 = nn.init.kaiming_normal_(self.wout1)
        # self.outputs.append(wout1)

    def get_wout(self):
        return self.wout,self.wout1


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
        y0 = torch.stack([y0 for _ in range(len(s))]).reshape(-1, 1)
        a1 = self.a1(t).reshape(-1, 1)
        a0 = self.a0(t).reshape(-1, 1)
        f = self.f(t).reshape(-1, 1)

        DH = (a1 * sd + a0 * s)
        D0 = (a0 * y0 - f).reshape(-1, 1)
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

        elif type == 'mask':
            self.g = lambda t: torch.tensor([0. if tt ==0 else 1. for tt in t])
            self.gd = lambda t: torch.tensor([0. if tt ==0 else 0. for tt in t])

    def get_g(self, t):
        return self.g(t)

    def get_gdot(self, t):
        return self.gd(t)

    def get_g_gdot(self,t):
        return self.g(t),self.gd(t)


def compute_s_sdot(func,zinit,batch_t,param):
    integ = odeint(func, zinit, batch_t.ravel(), method=args.method_rc)
    integdot = torch.stack([func(batch_t.ravel()[i], integ[i]) for i in range(len(integ))])

    integ = integ.reshape(-1, NDIMZ)
    integdot = integdot.reshape(-1, NDIMZ)

    bias = torch.ones((len(batch_t), 1)).to(device)
    integ = torch.cat([integ, bias], 1)

    bias2 = torch.zeros((len(batch_t), 1)).to(device)
    integdot = torch.cat([integdot, bias2], 1)

    gt = param.get_g(t).repeat_interleave(NDIMZ+1 ).reshape(-1, NDIMZ+1)
    gtd = param.get_gdot(t).repeat_interleave(NDIMZ+1).reshape(-1, NDIMZ+1 )

    s = gt * integ
    sd = gtd * integ + gt * integdot

    return s,sd


if args.viz:
    # makedirs('png')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr,s,sd,wout):
    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        for i in range(1):
            ax_traj.plot(true_y.cpu().numpy()[:, i, 0], true_y.cpu().numpy()[:, i, 1],
                         'g-')
            ax_traj.plot(pred_y.cpu().numpy()[:, i, 0], pred_y.cpu().numpy()[:, i, 1], '--', 'b--')


        ax_phase.plot(np.arange(s.shape[1]),s[-1].cpu().numpy())

        ax_vecfield.plot(np.arange((wout[0].shape[0])),wout[0].cpu().numpy(),np.arange((wout[0].shape[0])),wout[1].cpu().numpy())

        # ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        # ax_traj.set_ylim(-2, 2)
        ax_traj.legend()
        # plt.show()
        plt.draw()
        plt.pause(0.001)


def get_ham(q,p):

    return (q**2)/2 + SCALER*(q**4)/4 + (p**2)/2

if __name__ == '__main__':

    ii = 0
    NDIMZ = args.hidden_size

    diffeq_init = diffeq()
    gt_generator = base_diffeq(diffeq_init)

    t = torch.arange(0.,args.tmax,args.dt).reshape(-1,1).to(device)

    true_y0 = torch.tensor([[1.3, 1.]]).to(device)
    true_y = gt_generator.get_solution(true_y0,t.ravel())
    # print(true_y.shape)
    # print(get_ham(true_y[:,0,0],true_y[:,0,1]))

    #
    # wout generator
    if args.wout == 'analytic':
        wout_gen = Transformer_Analytic(a0, a1, f, 0.01)
    else:
        wout_gen = Transformer_Learned(NDIMZ,1,true_y0.shape[1]).to(device)
    # hidden state generator
    func = ODEFunc(NDIMZ).to(device)

    if args.wout  == 'analytic':
        optimizer = optim.Adam(func.parameters(),lr=1e-3)
    elif args.wout == 'learned':
        optimizer = optim.Adam([
            {'params': func.parameters(),     'lr': 1e-4 },
            {'params': wout_gen.parameters(), 'lr': 1e-2,'weight_decay': 1e-4}
        ])



    param = Parametrization(args.paramg)
    zinit = torch.zeros(1,NDIMZ).to(device)

    if args.evaluate_only is False:
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            s,sd = compute_s_sdot(func,zinit,t,param)


            if args.wout == 'analytic':
                wout = wout_gen.get_wout(s, sd, true_y0, t)
            elif args.wout == 'learned':
                wout = wout_gen.get_wout()

            pred_q = true_y0[0,0].reshape(1, 1) + torch.mm(s, wout[0])
            pred_qd = torch.mm(sd,wout[0])
            pred_p = true_y0[0,1].reshape(1,1) + torch.mm(s,wout[1])
            pred_pd = torch.mm(sd,wout[1])

            c1 = ((pred_qd-pred_p)**2)
            c2 = ((pred_pd + pred_q + SCALER*pred_q**3)**2)
            c3 = (get_ham(pred_q,pred_p).reshape(-1,1) - get_ham(true_y0[0,0],true_y0[0,1]).reshape(1,1))**2
            lst = c1 + c2  + c3
            loss = torch.mean(lst)

            loss.backward()
            optimizer.step()

            if itr % args.test_freq == 0:
                print(f'c1:{c1.mean().item()},c2:{c2.mean().item()},c3:{c3.mean().item()}')
                # print(loss.item())
                with torch.no_grad():
                    s, sd = compute_s_sdot(func, zinit, t, param)
                    if args.wout == 'analytic':
                        wout = wout_gen.get_wout(s, sd, true_y0, t)
                    elif args.wout == 'learned':
                        wout = wout_gen.get_wout()

                    pred_q = true_y0[0, 0].reshape(1, 1) + torch.mm(s, wout[0])
                    pred_p = true_y0[0, 1].reshape(1, 1) + torch.mm(s, wout[1])
                    pred_y = torch.cat([pred_q,pred_p],1).reshape(-1,1,2)
                    loss = torch.mean(torch.abs(pred_y - true_y))
                    print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                    visualize(true_y, pred_y, func, ii,s,sd,wout)
                    ii += 1



        ## inference ##
        torch.save(func.state_dict(), 'func_dict_wout1')
        torch.save(wout_gen.state_dict(), 'wout_dict_wout1')

    ii = 0
    func = ODEFunc(NDIMZ)
    func.load_state_dict(torch.load('func_dict_wout1'))
    func.eval()
    # freeze all weights
    for param_vals in func.parameters():
        param_vals.requires_grad = False

    s, sd = compute_s_sdot(func, zinit, t, param)
    q_ics = torch.linspace(1, 2., 20)
    p_ics = torch.linspace(1, 2., 20)
    ics = torch.cat([q_ics.reshape(-1,1),p_ics.reshape(-1,1)],1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 7))
    rmsr = 0.

    for ic in ics:
        wout_gen = Transformer_Learned(NDIMZ,1, 2)
        wout_gen.load_state_dict(torch.load('wout_dict_wout1'))
        optimizer = optim.Adam(wout_gen.parameters(), lr=1e-3)

        for itr in range(1, args.niters_wout + 1):
            optimizer.zero_grad()
            wout = wout_gen.get_wout()
            pred_q = ic[0].reshape(1, 1) + torch.mm(s, wout[0])
            pred_qd = torch.mm(sd, wout[0])
            pred_p = ic[1].reshape(1, 1) + torch.mm(s, wout[1])
            pred_pd = torch.mm(sd, wout[1])

            c3 = (get_ham(pred_q,pred_p).reshape(-1,1) - get_ham(ic[0],ic[1]).reshape(1,1))**2

            lst = (pred_qd - pred_p) ** 2 + (pred_pd + pred_q + SCALER*pred_q ** 3) ** 2  +c3# + energy term
            loss = torch.mean(lst)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            y0 = ic.reshape(1,2)
            true_y = gt_generator.get_solution(y0,t.ravel())
            wout = wout_gen.get_wout()

            pred_q = y0[0, 0].reshape(1, 1) + torch.mm(s, wout[0])
            pred_qd = torch.mm(sd, wout[0])
            pred_p = y0[0, 1].reshape(1, 1) + torch.mm(s, wout[1])
            pred_pd = torch.mm(sd, wout[1])

            pred_y = torch.cat([pred_q.reshape(-1,1),pred_p.reshape(-1,1)],1)
            rmsr += (pred_qd - pred_p) ** 2 + (pred_pd + pred_q + SCALER*pred_q ** 3) ** 2  # + energy term

            ax[0].plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1],
                         'g-')
            ax[0].plot(pred_y.cpu().numpy()[:, 0], pred_y.cpu().numpy()[:, 1], '--', 'b--')

            ax[0].set_ylabel('y(t)')
            ax[0].set_xlabel('t')
            residual = ((true_y - pred_y.reshape(-1,1,2))**2).cpu().numpy()
            ax[1].plot(t.cpu().numpy(),(residual.reshape(-1,2)).mean(1),'--')
            ax[1].set_yscale('log')
            ax[1].set_xlabel('time')
            ax[1].set_ylabel('MSE Error')

    with torch.no_grad():
        ax[2].plot(t.cpu().numpy(), np.sqrt(rmsr.reshape(-1, 1)/len(ics)), '--')
        ax[2].set_yscale('log')
        ax[2].set_xlabel('time')
        ax[2].set_ylabel('RMSR')
        plt.show()
