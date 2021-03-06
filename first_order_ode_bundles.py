"""
base solver for transfer ode (first order methods)
"""
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
import random
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *

torch.manual_seed(46)
parser = argparse.ArgumentParser('transfer demo')

parser.add_argument('--tmax', type=float, default=3.)
parser.add_argument('--dt', type=int, default=0.05)
parser.add_argument('--niters', type=int, default=3000)
parser.add_argument('--niters_test', type=int, default=15000)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--num_bundles', type=int, default=10)
parser.add_argument('--num_bundles_test', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_false')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--evaluate_only', action='store_true')
args = parser.parse_args()
scaler = MinMaxScaler()


torch.set_default_tensor_type('torch.DoubleTensor')


class diffeq(nn.Module):
    """
    defines the diffeq of interest
    """

    def __init__(self, a0, f):
        super().__init__()
        # self.a1 = a1
        self.a0 = a0
        self.f = f

    def forward(self, t, states):
        # print()
        y = states[:, 0].reshape(1, -1)
        yd = get_udot(t, y, self.a0,self.f)  # (-self.a1(t) * yd - self.a0(t) * y + self.f(t)).reshape(-1, 1)
        return yd.reshape(-1,1)

def get_udot(t,y,a,f):

    if y.shape[0] <=1:
        a0 = torch.tensor([a_(t) for a_ in a]).reshape(1,-1)
        f0 = torch.tensor([f_(t) for f_ in f]).reshape(1,-1)
    else:
        a0 = torch.cat([a_(t.reshape(-1,1)) for a_ in a],1)
        f0 = torch.cat([f_(t.reshape(-1,1)) for f_ in f],1)

    yd = (-a0 * y + f0)
    return yd


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


def get_wout(s, sd, y0, t,a0s,fs):
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
        DH = (a1 * sd + a0 * s)
        h0m = s[0].reshape(-1, 1)

        W0 = torch.linalg.solve(DH.t() @ DH + h0m @ h0m.t(), DH.t() @ D0 + h0m @ (y0[0, :].reshape(1, -1)))
        WS.append(W0)

    nWS = (torch.cat(WS)).reshape(nf.shape[1], -1)
    return nWS.t()




if args.viz:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, lst):
    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        for i in range(args.num_bundles):
            ax_traj.plot(t.detach().cpu().numpy(), true_y.cpu().numpy()[:, i],
                         'g-')
            ax_traj.plot(t.detach().cpu().numpy(), pred_y.cpu().numpy()[:, i], '--', 'b--')
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
    #need to sample tuple of (a1,f,IC)
    # each column of Wouts defines a solution thus, each tuple defines a solution too


    f_train = [lambda t: torch.cos(t),lambda t: torch.sin(t),lambda t: 1*t]
    a0_train = [lambda t: t,lambda t:t**2, lambda t: 1*t]
    r1 = -5.
    r2 = 5.
    true_y0 = (r2 - r1) * torch.rand(100) + r1
    t = torch.arange(0., args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True

    # sample each parameter to build the tuples
    f_samples = random.choices(f_train, k=args.num_bundles)
    a0_samples = random.choices(a0_train, k=args.num_bundles)
    y0_samples = torch.tensor(random.choices(true_y0, k=args.num_bundles)).reshape(1,-1)

    diffeq_init = diffeq(a0_samples,f_samples)
    gt_generator = base_diffeq(diffeq_init)
    true_y = gt_generator.get_solution(y0_samples.reshape(-1,1),t.ravel()).reshape(-1,args.num_bundles)

    # use this quick test to find gt solutions and check training ICs
    # have a solution (don't blow up for dopri5 integrator)
    # true_y = gt_generator.get_solution(true_y0.reshape(-1, 1), t.ravel())

    # instantiate wout with coefficients
    func = ODEFunc(hidden_dim=NDIMZ, output_dim=args.num_bundles)

    optimizer = optim.Adam(func.parameters(), lr=1e-3)

    loss_collector = []

    best_residual = 1e-1

    if not args.evaluate_only:

        for itr in range(1, args.niters + 1):
            s1 = time.time()
            func.train()

            # add t0 to training times, including randomly generated ts
            t0 = torch.tensor([[0.]])
            t0.requires_grad = True
            tv = args.tmax * torch.rand(50).reshape(-1, 1)
            tv.requires_grad = True
            tv = torch.cat([t0, tv], 0)
            optimizer.zero_grad()

            # compute hwout,hdotwout
            pred_y = func(tv)
            pred_ydot = diff(pred_y, tv)

            # enforce diffeq
            loss_diffeq = pred_ydot - get_udot(tv,pred_y,a0_samples,f_samples)

            # enforce initial conditions
            loss_ics = pred_y[0, :].ravel() - y0_samples.ravel()

            loss = torch.mean(torch.square(loss_diffeq)) + torch.mean(torch.square(loss_ics))
            loss.backward()
            optimizer.step()
            loss_collector.append(torch.square(loss_diffeq).mean().item())

            # print(time.time()-s1)
            if itr % args.test_freq == 0:
                func.eval()
                pred_y = func(t)
                pred_ydot = diff(pred_y,t)

                pred_y = pred_y.detach()
                pred_ydot = pred_ydot.detach()

                visualize(true_y.detach(), pred_y.detach(), loss_collector)
                ii += 1

                current_residual = torch.mean((pred_ydot - get_udot(t,pred_y,a0_samples,f_samples))**2)
                if current_residual < best_residual:
                    torch.save(func.state_dict(), 'func_ffnn_bundles')
                    best_residual = current_residual
                    print(itr,best_residual.item())

    # with torch.no_grad():

    # f_test = [lambda t: torch.sin(t)]
    # # keep fixed to one list element - else need to do tensor math to compute Wout
    # a0_test = [lambda t: t**3]
    # r1 = -15.
    # r2 = 15.
    # true_y0 = (r2 - r1) * torch.rand(100) + r1
    # t = torch.arange(0., args.tmax, args.dt).reshape(-1, 1)
    # t.requires_grad = True
    #
    # # sample each parameter to build the tuples
    # f_samples = random.choices(f_test, k=args.num_bundles_test)
    # a0_samples = random.choices(a0_test, k=args.num_bundles_test)
    # y0_samples = torch.tensor(random.choices(true_y0, k=args.num_bundles_test)).reshape(1, -1)
    #
    # # print(y0_samples.shape)
    # diffeq_init = diffeq(a0_samples, f_samples)
    # gt_generator = base_diffeq(diffeq_init)
    #
    #
    # func.load_state_dict(torch.load('func_ffnn_bundles'))
    # func.eval()
    #
    # h = func.h(t)
    # hd = diff(h, t)
    # h = h.detach()
    # hd = hd.detach()
    #
    #
    # plt.figure()
    #
    # plt.plot(h)
    # plt.show()
    #
    #
    #
    # gz_np = h.detach().numpy()
    # T = np.linspace(0, 1, len(gz_np)) ** 2
    # new_hiddens = scaler.fit_transform(gz_np)
    # pca = PCA(n_components=3)
    # pca_comps = pca.fit_transform(new_hiddens)
    #
    # fig = plt.figure()
    # # ax = plt.axes(projection='3d')
    #
    # if pca_comps.shape[1] >= 2:
    #     s = 10  # Segment length
    #     for i in range(0, len(gz_np) - s, s):
    #         plt.plot(pca_comps[i:i + s + 1, 0], pca_comps[i:i + s + 1, 1])
    #     # s = 10  # Segment length
    #     # for i in range(0, len(gz_np) - s, s):
    #     #     ax.plot3D(pca_comps[i:i + s + 1, 0], pca_comps[i:i + s + 1, 1], pca_comps[i:i + s + 1, 2],
    #     #               color=(0.1, 0.8, T[i]))
    #     #     plt.xlabel('comp1')
    #     #     plt.ylabel('comp2')
    #
    #
    # s1 = time.time()
    # wout = get_wout(h, hd, y0_samples, t.detach(),a0_samples[0],f_samples)
    # pred_y = h @ wout
    # s2 = time.time()
    # print(f'all_ics:{s2 - s1}')
    #
    # s1 = time.time()
    # true_ys = (gt_generator.get_solution(y0_samples, t.ravel())).reshape(-1, args.num_bundles_test)
    # s2 = time.time()
    # print(f'gt_ics:{s2 - s1}')
    #
    # print(true_ys.shape,pred_y.shape)
    #
    # # s1 = time.time()
    # # true_y = estim_generator.get_solution(ics.reshape(-1, 1), t.ravel())
    # # estim_ys = true_y.reshape(len(pred_y), ics.shape[1])
    # # s2 = time.time()
    # # print(f'estim_ics:{s2 - s1}')
    #
    # print(f'prediction_accuracy:{((pred_y - true_ys) ** 2).mean()} pm {((pred_y - true_ys) ** 2).std()}')
    # # print(f'estim_accuracy:{((estim_ys - true_ys) ** 2).mean()} pm {((estim_ys - true_ys) ** 2).std()}')
    #
    # fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # # print(true_ys[0,:])
    # for i in range(0, args.num_bundles_test, 50):
    #     ax[0].plot(t.detach().cpu().numpy(), true_ys.cpu().numpy()[:, i], c='blue', linestyle='dashed')
    #     ax[0].plot(t.detach().cpu().numpy(), pred_y.cpu().numpy()[:, i], c='orange')
    #     # plt.draw()
    #
    # ax[1].plot(t.detach().cpu().numpy(), ((true_ys - pred_y) ** 2).mean(1).cpu().numpy(), c='green')
    # ax[1].set_xlabel('Time (s)')
    # plt.legend()
    # plt.show()

    func.load_state_dict(torch.load('func_ffnn_bundles'))
    func.eval()

    # pred_y = func(t)
    # pred_ydot = diff(pred_y, t)
    # pred_yddot = diff(pred_ydot, t)
    #
    # pred_y = pred_y.detach()
    # pred_ydot = pred_ydot.detach()
    # pred_yddot = pred_yddot.detach()

    sns.axes_style(style='ticks')
    sns.set_context("paper", font_scale=2.3,
                    rc={"font.size": 30, "axes.titlesize": 25, "axes.labelsize": 20, "axes.legendsize": 20,
                        'lines.linewidth': 2.5})
    sns.set_palette('deep')
    sns.set_color_codes(palette='deep')

    f_train = [lambda t: torch.cos(t), lambda t: torch.sin(t), lambda t: 1 * t]
    a0_train = [lambda t: t, lambda t: t ** 2, lambda t: 1 * t]
    r1 = -5.
    r2 = 5.
    true_y0 = (r2 - r1) * torch.rand(100) + r1
    t = torch.arange(0., args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True

    # sample each parameter to build the tuples
    f_samples = random.choices(f_train, k=args.num_bundles_test)
    a0_samples = random.choices(a0_train, k=args.num_bundles_test)
    y0_samples = torch.tensor(random.choices(true_y0, k=args.num_bundles_test)).reshape(args.num_bundles_test,1)

    diffeq_init = diffeq(a0_samples, f_samples)
    gt_generator = base_diffeq(diffeq_init)


    print(y0_samples.shape)
    true_y = gt_generator.get_solution(y0_samples, t.ravel())

    # diffeq_init = diffeq(a0_samples, f_samples)
    # gt_generator = base_diffeq(diffeq_init)
    # true_y = gt_generator.get_solution(y0_samples, t.ravel())
    # print(true_y.shape)
    h = func.h(t)
    hd = diff(h, t)
    # hdd = diff(hd, t)
    h = h.detach()
    hd = hd.detach()
    # hdd = hdd.detach()

    h = torch.cat([h,torch.ones(len(h),1)],1)
    hd = torch.cat([hd,torch.zeros(len(hd),1)],1)
    # hdd = torch.cat([hdd,torch.zeros(len(hdd),1)],1)

    s1 = time.time()

    wout = get_wout(h, hd, y0_samples, t.detach(), a0_samples, f_samples)
    print(f'wout:{time.time()-s1}')

    with torch.no_grad():
        pred_y = h@wout
        pred_yd = hd@wout
        # pred_ydd = hdd @ wout
        current_residual = torch.mean((pred_yd - get_udot(t, pred_y, a0_samples, f_samples)) ** 2,1)
        current_err = torch.std((pred_yd - get_udot(t, pred_y, a0_samples, f_samples)) ** 2,1)
        # print(current_residual)

        # print(f'prediction_accuracy:{((pred_y - true_y) ** 2).mean()} pm {((pred_y - true_y) ** 2).std()}')

        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(6, 8))

        for i in range(10):
            a0.plot(pred_y.cpu().numpy()[:, i], linewidth=5)
            a0.plot(true_y.cpu().numpy()[:, i, 0], linestyle='--', color='black')

        a0.set_xlabel(r'$u$')
        a0.set_ylabel(r'$\dot{u}$')


        # residual = ((pred_y.cpu().numpy()[:, :] - true_y.cpu().numpy()[:, :, 0])**2 + (true_y.cpu().numpy()[:, :, 1]-pred_yd.cpu().numpy()[:, :])**2)/2.
        # print(residual.shape,residual.min())
        a1.set_yscale('log')
        a1.plot(np.arange(len(current_residual))*(args.tmax/len(current_residual)),current_residual.numpy(),c='royalblue')
        xv = np.arange(len(current_residual))*(args.tmax/len(current_residual))

        # a1.fill_between(xv, current_residual-current_err, current_residual+current_err)
        print(torch.mean(current_residual))
        print(torch.std(current_residual))

        a1.set_xlabel('Time (s)')
        a1.set_ylabel('Residuals')
        plt.tight_layout()
        print('yes')
        plt.savefig('results_first_test.pdf',dpi=2400,bbox_inches='tight')
        print('yes')
