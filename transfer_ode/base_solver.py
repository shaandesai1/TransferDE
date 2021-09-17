"""
base solver for transfer ode
"""
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import numpy as np

ADJOINT = False
if ADJOINT:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

hi = "bye"

class diffeq(nn.Module):
    """
    defines the diffeq of interest

    Arguments:
    ----------
    a0:

    a1:

    f:
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

    #what is z0?
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
        return self.wout, 0


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

    def _center_data(self, X, y):
        """
        INSTRUCTIONS:
        1. assign `_x_means` to self, along the axis such that 
           the numbers of means matches the number of features (2)
        2. assign `_y_mean` to self (y.mean())
        3. subtract _x_means from X and assign it to X_centered
        4. subtract _y_mean from y and assign it to y_centered
        """
        self._x_means = X.mean(axis=0)
        self._y_means = y.mean(axis = 0)

        X_centered = X - self._x_means
        y_centered = y - self._y_means

        return X_centered, y_centered

    def calc_bias(self, weights):
        return self._y_means - self._x_means @ weights

    # def get_wout(self, s, sd, y0, t):
    #     y0 = torch.stack([y0 for _ in range(len(s))]).reshape(len(s),-1)
    #     a1 = self.a1(t).reshape(-1, 1)
    #     a0 = self.a0(t).reshape(-1, 1)
    #     f = self.f(t).reshape(-1, 1)

    #     DH = (a1 * sd + a0 * s)
    #     D0 = (a0 * y0 - f)
    #     lambda_0 = self.lambda_
    #     W0 = torch.linalg.solve(DH.t() @ DH + lambda_0, -DH.t() @ D0)
    #     return W0, 0
    def _center_H(self, inputs = None, outputs = None, keep = False):
        """
        INSTRUCTIONS:
        1. assign `_x_means` to self, along the axis such that 
           the numbers of means matches the number of features (2)
        2. assign `_y_mean` to self (y.mean())
        3. subtract _x_means from X and assign it to X_centered
        4. subtract _y_mean from y and assign it to y_centered
        """
        if inputs is not None:
            X = inputs

            if keep:
                self._x_means = X.mean(axis=0)
                self._x_stds = X.std(axis = 0)

            X_centered = (X - self._x_means)/self._x_stds
            return X_centered
        if outputs is not None:
            y = outputs

            if keep:
                self._y_means = y.mean(axis = 0)

            y_centered = y - self._y_means #(y - y_means)/y_stds
            return y_centered

    def get_wout(self, s, sd, y0, t):
        y0 = torch.stack([y0 for _ in range(len(s))]).reshape(len(s),-1)
        a1 = self.a1(t).reshape(-1, 1)
        a0 = self.a0(t).reshape(-1, 1)
        f = self.f(t).reshape(-1, 1)

        DH = (a1 * sd + a0 * s)


        _ = self._center_H(inputs = -DH, keep = True)


        D0 = (a0 * y0 - f)
        lambda_0 = self.lambda_
        W0 = torch.linalg.solve(DH.t() @ DH + lambda_0, -DH.t() @ D0)

        
        _  = self._center_H(outputs = D0, keep = True)

        bias = self.calc_bias(W0)
        return W0, bias

        # #with torch.no_grad():
        # y0 = torch.stack([y0 for _ in range(len(s))]).reshape(len(s),-1)
        # a1 = self.a1(t).reshape(-1, 1)
        # a0 = self.a0(t).reshape(-1, 1)
        # f = self.f(t).reshape(-1, 1)

        # # no apparent consideration of the bias. This would case it to fail?
        # DH = (a1 * sd + a0 * s)

        # #ones_col = torch.ones_like(DH[:,0]).reshape(-1,1)
        # #DH = torch.hstack((ones_col, DH))
        # #assert False, DH.mean(axis = 0)
        # D0 = (a0 * y0 - f)


        # lambda_0 = self.lambda_
        # #xx, yy = self._center_data(DH, D0)
        # DH1 = DH.T @ DH 
        # DH2 = DH1 + lambda_0 * torch.eye(DH1.shape[1], dtype = torch.float32)
        # W0 = torch.linalg.solve( - DH2,  DH.T @ D0)
        # bias = W0[0] 
        # weight = W0[1:]
        # #W0 = torch.linalg.solve(DH1, )
        # #bias =  self.calc_bias(W0) #yy.mean()#
        # return weight, bias


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


def visualize(true_y, pred_y, odefunc, itr,lst,s):



    if args.viz:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 4), facecolor='white')
        ax_traj = fig.add_subplot(131, frameon=False)
        ax_phase = fig.add_subplot(132, frameon=False)
        ax_vecfield = fig.add_subplot(133, frameon=False)
        plt.show(block=False)

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
