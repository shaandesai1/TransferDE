"""
base solver for transfer ode (first order methods). The code here is an example of how to solve
first order ODEs. It trains a PINN on 'num_bundles' equations simultaneously. Then, the network
weights are frozen and the hidden layers are used as a basis function to solve new unseen equations.
"""
import argparse
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

from utils import *

torch.manual_seed(46)
parser = argparse.ArgumentParser("first order transfer demo")

parser.add_argument(
    "--tmax", type=float, default=3.0
)  # maximum time to solve equation for (0,T_max) is the range
parser.add_argument(
    "--dt", type=int, default=0.05
)  # timestep between points (even sampling)
parser.add_argument("--niters", type=int, default=3000)  # number of training iterations
parser.add_argument("--niters_test", type=int, default=15000)
parser.add_argument("--hidden_size", type=int, default=100)  # hidden nodes size
parser.add_argument("--num_bundles", type=int, default=10)  # number of training bundles
parser.add_argument(
    "--num_bundles_test", type=int, default=1000
)  # number of equations to test with frozen hidden layers
parser.add_argument("--test_freq", type=int, default=20)
parser.add_argument("--viz", action="store_false")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--evaluate_only", action="store_true")
args = parser.parse_args()
scaler = MinMaxScaler()


torch.set_default_tensor_type("torch.DoubleTensor")


def get_udot(t, y, a, f):
    """
    Given time 't', initial state 'y' and functions a(t) and f(t) return
    dy/dt = -a(t)*y + f(t)
    Args:
        t: input time vector
        y: initial state y_0 of the system
        a: homogenous coefficient function in the diffeq
        f: external forcing function of the diffeq
    """
    if y.shape[0] <= 1:
        a0 = torch.tensor([a_(t) for a_ in a]).reshape(1, -1)
        f0 = torch.tensor([f_(t) for f_ in f]).reshape(1, -1)
    else:
        a0 = torch.cat([a_(t.reshape(-1, 1)) for a_ in a], 1)
        f0 = torch.cat([f_(t.reshape(-1, 1)) for f_ in f], 1)

    yd = -a0 * y + f0
    return yd


class diffeq(nn.Module):
    """
    defines the differential equation of interest
    """

    def __init__(self, a0, f):
        """
        Function that takes the a0 and f functions
        """
        super().__init__()
        self.a0 = a0
        self.f = f

    def forward(self, t, states):
        """
        when called, this function returns the dy/dt value
        of dy/dt = -a0(t)y(t) + f(t) given time 't' and state [y_0,y'_0]

        Args:
            t: time inputs
            states: initial position and velocity [y_0,y'_0]
        """
        y = states[:, 0].reshape(1, -1)
        yd = get_udot(t, y, self.a0, self.f)
        return yd.reshape(-1, 1)



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


def get_wout(s, sd, y0, t, a0s, fs):
    """
    Function to compute the weights W_out given hidden layers s and sd.
    Args:
        s: hidden layer of the network
        sd: first derivative of the hidden layer of the network
        y0: initial condition
        t: time vector
        a0s: samples of the homogenous coefficient function in dy/dt = -a0*y +f
        fs: samples of the forcing function in dy/dt = -a0*y + f
    """
    ny0 = torch.stack([y0 for _ in range(len(s))]).reshape(len(s), -1)
    na0 = torch.cat([a_(t) for a_ in a0s], 1)
    na1 = torch.ones_like(na0)
    nf = torch.cat([f_(t) for f_ in fs], 1)

    WS = []
    for i in range(nf.shape[1]):
        y0 = ny0[:, i].reshape(-1, 1)
        a0 = na0[:, i].reshape(-1, 1)
        a1 = na1[:, i].reshape(-1, 1)
        f = nf[:, i].reshape(-1, 1)

        D0 = f
        DH = a1 * sd + a0 * s
        h0m = s[0].reshape(-1, 1)

        W0 = torch.linalg.solve(
            DH.t() @ DH + h0m @ h0m.t(), DH.t() @ D0 + h0m @ (y0[0, :].reshape(1, -1))
        )
        WS.append(W0)

    nWS = (torch.cat(WS)).reshape(nf.shape[1], -1)
    return nWS.t()


if args.viz:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), facecolor="white")
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, lst):
    """
    Plot the ground truth as dashed lines and the predicted in color.
    """
    if args.viz:
        ax_traj.cla()
        ax_traj.set_title("Trajectories")
        ax_traj.set_xlabel("t")
        ax_traj.set_ylabel("x,y")
        for i in range(args.num_bundles):
            ax_traj.plot(t.detach().cpu().numpy(), true_y.cpu().numpy()[:, i], "g-")
            ax_traj.plot(
                t.detach().cpu().numpy(), pred_y.cpu().numpy()[:, i], "--", "b--"
            )
        ax_phase.set_yscale("log")
        ax_phase.plot(np.arange(len(lst)), lst)

        ax_traj.legend()

        plt.draw()
        plt.pause(0.001)


if __name__ == "__main__":
    ii = 0
    NDIMZ = args.hidden_size
    # we want to solve first order differential equations
    # to do so, we write them down as dy/dt = -a(t)y + f(t)
    # in some places we write a(t) as a_0 so apols for the confusion, same with f(t) and f_0
    # ofcourse, to train and test we need MANY equations
    # to get many equations, we ideally want to sample them so we define a set of options for
    # a(t), f(t) and the initial conditions
    # a combination of one from each, defines a specific equation.
    # by sampling from f_train, a0_train and the initial conditions below we get a tuple (a0,f,IC) that defines a
    # differential equation.

    f_train = [lambda t: torch.cos(t), lambda t: torch.sin(t), lambda t: 1 * t]
    a0_train = [lambda t: t, lambda t: t**2, lambda t: 1 * t]
    r1 = -5.0
    r2 = 5.0
    true_y0 = (r2 - r1) * torch.rand(100) + r1
    t = torch.arange(0.0, args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True

    # sample each parameter to build the tuples
    f_samples = random.choices(f_train, k=args.num_bundles)
    a0_samples = random.choices(a0_train, k=args.num_bundles)
    y0_samples = torch.tensor(random.choices(true_y0, k=args.num_bundles)).reshape(
        1, -1
    )
    # for every condition sampled, find the ground truth solution using a numerical integrator
    diffeq_init = diffeq(a0_samples, f_samples)
    gt_generator = base_diffeq(diffeq_init)
    true_y = gt_generator.get_solution(y0_samples.reshape(-1, 1), t.ravel()).reshape(
        -1, args.num_bundles
    )

    # # instantiate wout with coefficients
    func = ODEFunc(hidden_dim=NDIMZ, output_dim=args.num_bundles)

    optimizer = optim.Adam(func.parameters(), lr=1e-3)

    loss_collector = []

    best_residual = 1e-1

    ### TRAINING OF THE PINNs ###
    if not args.evaluate_only:
        for itr in range(1, args.niters + 1):
            s1 = time.time()
            func.train()

            # add t0 to training times, including randomly generated ts
            t0 = torch.tensor([[0.0]])
            t0.requires_grad = True
            tv = args.tmax * torch.rand(50).reshape(-1, 1)
            tv.requires_grad = True
            tv = torch.cat([t0, tv], 0)
            optimizer.zero_grad()

            # compute hwout,hdotwout
            pred_y = func(tv)
            pred_ydot = diff(pred_y, tv)

            # enforce diffeq
            loss_diffeq = pred_ydot - get_udot(tv, pred_y, a0_samples, f_samples)

            # enforce initial conditions
            loss_ics = pred_y[0, :].ravel() - y0_samples.ravel()

            loss = torch.mean(torch.square(loss_diffeq)) + torch.mean(
                torch.square(loss_ics)
            )
            loss.backward()
            optimizer.step()
            loss_collector.append(torch.square(loss_diffeq).mean().item())

            # print(time.time()-s1)
            if itr % args.test_freq == 0:
                func.eval()
                pred_y = func(t)
                pred_ydot = diff(pred_y, t)

                pred_y = pred_y.detach()
                pred_ydot = pred_ydot.detach()

                visualize(true_y.detach(), pred_y.detach(), loss_collector)
                ii += 1

                current_residual = torch.mean(
                    (pred_ydot - get_udot(t, pred_y, a0_samples, f_samples)) ** 2
                )
                if current_residual < best_residual:
                    torch.save(func.state_dict(), "func_ffnn_bundles") #saves the best model
                    best_residual = current_residual
                    print(itr, best_residual.item())


    ### EVALUATE THE TRAINED PINN ###
    func.load_state_dict(torch.load("func_ffnn_bundles"))
    func.eval()

    sns.axes_style(style="ticks")
    sns.set_context(
        "paper",
        font_scale=2.3,
        rc={
            "font.size": 30,
            "axes.titlesize": 25,
            "axes.labelsize": 20,
            "axes.legendsize": 20,
            "lines.linewidth": 2.5,
        },
    )
    sns.set_palette("deep")
    sns.set_color_codes(palette="deep")

    # define the f's, a's and initial conditions you want to sample your test from
    f_train = [lambda t: torch.cos(t), lambda t: torch.sin(t), lambda t: 1 * t]
    a0_train = [lambda t: t, lambda t: t**2, lambda t: 1 * t]
    r1 = -5.0
    r2 = 5.0
    true_y0 = (r2 - r1) * torch.rand(100) + r1
    t = torch.arange(0.0, args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True

    # sample each parameter to build the tuples
    f_samples = random.choices(f_train, k=args.num_bundles_test)
    a0_samples = random.choices(a0_train, k=args.num_bundles_test)
    y0_samples = torch.tensor(random.choices(true_y0, k=args.num_bundles_test)).reshape(
        args.num_bundles_test, 1
    )

    diffeq_init = diffeq(a0_samples, f_samples)
    gt_generator = base_diffeq(diffeq_init)

    print(y0_samples.shape)
    true_y = gt_generator.get_solution(y0_samples, t.ravel())

    # compute the hidden layers from the function
    h = func.h(t)
    hd = diff(h, t) #hd implies h_dot
    h = h.detach()
    hd = hd.detach()

    h = torch.cat([h, torch.ones(len(h), 1)], 1)
    hd = torch.cat([hd, torch.zeros(len(hd), 1)], 1)

    s1 = time.time()

    #this function computes the W coefficients of the linear problem given the hidden layers
    #and conditions of the equations
    wout = get_wout(h, hd, y0_samples, t.detach(), a0_samples, f_samples)
    print(f"wout:{time.time()-s1}")

    with torch.no_grad():
        pred_y = h @ wout
        pred_yd = hd @ wout
        current_residual = torch.mean(
            (pred_yd - get_udot(t, pred_y, a0_samples, f_samples)) ** 2, 1
        )
        current_err = torch.std(
            (pred_yd - get_udot(t, pred_y, a0_samples, f_samples)) ** 2, 1
        )

        f, (a0, a1) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(6, 8)
        )

        for i in range(10):
            a0.plot(pred_y.cpu().numpy()[:, i], linewidth=5)
            a0.plot(true_y.cpu().numpy()[:, i, 0], linestyle="--", color="black")

        a0.set_xlabel(r"$u$")
        a0.set_ylabel(r"$\dot{u}$")

        a1.set_yscale("log")
        a1.plot(
            np.arange(len(current_residual)) * (args.tmax / len(current_residual)),
            current_residual.numpy(),
            c="royalblue",
        )
        xv = np.arange(len(current_residual)) * (args.tmax / len(current_residual))

        print(torch.mean(current_residual))
        print(torch.std(current_residual))

        a1.set_xlabel("Time (s)")
        a1.set_ylabel("Residuals")
        plt.tight_layout()
        plt.savefig("results_first_test.pdf", dpi=2400, bbox_inches="tight")
