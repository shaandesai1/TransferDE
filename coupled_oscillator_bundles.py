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
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['text.usetex'] = True

parser = argparse.ArgumentParser('transfer demo')

parser.add_argument('--tmax', type=float, default=10.)
parser.add_argument('--dt', type=int, default=0.01)
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--niters_test', type=int, default=15000)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--num_bundles', type=int, default=10)
parser.add_argument('--num_bundles_test', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=200)
parser.add_argument('--viz', action='store_false')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--evaluate_only', action='store_true')
args = parser.parse_args()
scaler = MinMaxScaler()

torch.set_default_tensor_type('torch.DoubleTensor')

# print(args.evaluate_only==False)

class diffeq(nn.Module):
    """
    defines the diffeq of interest
    """

    def __init__(self):
        super().__init__()
        # self.a1 = a1
        self.Amatrix = torch.tensor([[0,1],[-1,0]])
    # return ydot
    def forward(self, t, y):
        return get_udot(y)

def get_udot(y):
    # Amatrix = torch.tensor([[0., 1.], [-1., 0.]])
    # yd = Amatrix @ y.t()
    # return yd.t()
    m1,m2 = 1.,1.
    k1,k2 = 2.,4.
    Lmat = torch.tensor([[m1, 0.], [0., m2]])
    Rmat = torch.tensor([[k1 + k2, -k2], [-k2, k1 + k2]])
    Amatrix = Rmat
    yd = -Amatrix @ y.t()
    return yd.t()



class ODEFunc(nn.Module):
    """
    function to learn the outputs u(t) and hidden states h(t) s.t. u(t) = h(t)W_out
    """

    def __init__(self, hidden_dim, output_dim):
        super(ODEFunc, self).__init__()
        self.hdim = hidden_dim
        self.nl =  SiLU()
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


def get_wout(s, sd,sdd, y0,y0dot,m1,m2,k1,k2, t):


    Lmat = torch.tensor([[m1, 0.], [0., m2]])
    Rmat = torch.tensor([[k1 + k2, -k2], [-k2, k1 + k2]])
    Amatrix = torch.linalg.inv(Lmat)@Rmat
    hddothat = torch.block_diag(sdd,sdd)
    hdothat = torch.block_diag(sd,sd)
    hhat = torch.block_diag(s,s)
    Amatrixhat = torch.zeros((hdothat.shape[0],hdothat.shape[0]))
    for i in range(Amatrix.shape[0]):
        for j in range(Amatrix.shape[1]):
            Amatrixhat[i*s.shape[0]:(i+1)*s.shape[0],j*s.shape[0]:(j+1)*s.shape[0]]=torch.eye(s.shape[0],s.shape[0])*Amatrix[i,j]

    DH = hddothat + Amatrixhat@hhat


    h0= torch.block_diag(s[0,:].reshape(1,-1),s[0,:].reshape(1,-1))

    h0dot = torch.block_diag(sd[0, :].reshape(1, -1), sd[0, :].reshape(1, -1))

    W0 = torch.linalg.solve(DH.t()@DH + h0.t()@h0 + h0dot.t()@h0dot, h0.t()@y0.t() + h0dot.t()@y0dot.t() )
    return W0



if args.viz:


    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(pred_y,pred_yd, lst):
    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        for i in range(0,2*args.num_bundles,2):
            # ax_traj.plot(t.detach().cpu().numpy(), true_y.cpu().numpy()[:, i],
            #              'g-')
            ax_traj.plot( pred_y.cpu().numpy()[:, i], pred_yd.cpu().numpy()[:, i])
            ax_vecfield.plot(pred_y.cpu().numpy()[:,i])
            ax_vecfield.plot(pred_yd.cpu().numpy()[:,i])

        ax_phase.set_yscale('log')
        ax_phase.plot(np.arange(len(lst)), lst)

        ax_traj.legend()

        plt.draw()
        plt.pause(0.001)


def get_block_m(m1,m2):
    Amatrix = []

    # print(m1.shape)
    m1 = m1.reshape(1,-1)
    m2 = m2.reshape(1,-1)
    # print(m1[0,0])
    for i in range(m1.shape[1]):
        Amatrix.append(torch.tensor([[m1[0,i].item(), 0.], [0., m2[0,i].item()]]))

    # print(torch.block_diag(*Amatrix).shape)
    return torch.block_diag(*Amatrix)

def get_block_k(k1,k2):
    Amatrix = []

    k1 = k1.reshape(1,-1)
    k2 = k2.reshape(1,-1)
    for i in range(k1.shape[1]):
        Amatrix.append(torch.tensor([[k1[0,i]+k2[0,i], -k2[0,i]], [-k2[0,i], k1[0,i]+k2[0,i]]]))

    return torch.block_diag(*Amatrix)



def get_m(x_in,m1,m2):
    Amatrix = torch.tensor([[m1, 0.], [0., m2]])
    output = Amatrix @ x_in.t()
    return output.t()

def get_k(x_in,k1,k2):
    Amatrix = torch.tensor([[k1+k2, -k2], [-k2, k1+k2]])
    output = Amatrix @ x_in.t()
    return output.t()



if __name__ == '__main__':

    ii = 0
    NDIMZ = args.hidden_size

    r2 = 1.5
    r1 = -1.5

    #true_y0 = (r2 - r1) * torch.rand(2) + r1
    true_y0 = (r2 - r1) * torch.rand(100,2) + r1#torch.tensor([1.,1.]).reshape(1,2)
    true_y0dot = torch.zeros(100,2)#(r2 - r1) * torch.rand(100,2) + r1#torch.tensor([1., 3.]).reshape(1, 2)
    k1s = torch.linspace(0.5, 4.5, 100)#torch.ones(100)*0.5
    k2s = torch.linspace(.5, 4.5, 100)
    m1s = torch.linspace(1, 2, 100)
    m2s = torch.linspace(1, 2, 100)

    indices = random.choices(np.arange(100),k=args.num_bundles)
    # print(indices)
    y0s = true_y0[indices]
    # print(y0s)
    y0ds = true_y0dot[indices]
    k1sample = k1s[indices]
    k2sample = k2s[indices]
    m1sample = m1s[indices]
    m2sample = m2s[indices]

    # print(y0s,y0ds,k1sample,k2sample,m1sample,m2sample)

    t = torch.arange(0., args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True

    Mblock = get_block_m(m1sample,m2sample)
    Kblock = get_block_k(k1sample, k2sample)


    # instantiate wout with coefficients
    func = ODEFunc(hidden_dim=NDIMZ, output_dim=2*args.num_bundles)

    optimizer = optim.Adam(func.parameters(), lr=1e-3)

    loss_collector = []

    best_residual = 1e-1

    if not args.evaluate_only:

        for itr in range(1, args.niters + 1):
            func.train()

            # add t0 to training times, including randomly generated ts
            t0 = torch.tensor([[0.]])
            t0.requires_grad = True
            tv = args.tmax * torch.rand(int(args.tmax / args.dt))[:50].reshape(-1, 1)
            tv.requires_grad = True
            tv = torch.cat([t0, tv], 0)
            optimizer.zero_grad()

            # compute hwout,hdotwout


            pred_y = func(tv)
            # print('a')
            pred_ydot = diff(pred_y,tv)
            # print('b')
            pred_yddot = diff(pred_ydot, tv,1)
            # print('c')
            # enforce diffeq
            loss_diffeq = (Mblock@pred_yddot.t()).t() + (Kblock@pred_y.t()).t()
            # print('d')
            # loss_diffeq = (a1(tv.detach()).reshape(-1, 1)) * pred_ydot + (a0(tv.detach()).reshape(-1, 1)) * pred_y - f(
            #     tv.detach()).reshape(-1, 1)

            # enforce initial conditions
            loss_ics = (pred_y[0, :].ravel() - y0s.ravel()) + (pred_ydot[0, :].ravel() - y0ds.ravel())

            loss = torch.mean(torch.square(loss_diffeq)) + torch.mean(torch.square(loss_ics))
            loss.backward()
            optimizer.step()
            loss_collector.append(torch.square(loss_diffeq).mean().item())
            # print(loss_collector[-1])
            if itr % args.test_freq == 0:

                func.eval()
                pred_y = func(t)
                pred_ydot = diff(pred_y,t)
                pred_yddot = diff(pred_ydot, t)


                loss_diffeq = torch.mean(torch.square((Mblock @ pred_yddot.t()).t() + (Kblock @ pred_y.t()).t()))
                print(itr,loss_diffeq.item())
                if loss_diffeq.item() < best_residual:
                    torch.save(func.state_dict(), 'func_ffnn_systems_coupled')
                    best_residual = loss_diffeq.item()



    diffeq_init = diffeq()
    gt_generator = base_diffeq(diffeq_init)


    func.load_state_dict(torch.load('func_ffnn_systems_coupled'))
    func.eval()

    t = torch.arange(0., args.tmax, args.dt).reshape(-1, 1)
    t.requires_grad = True

    h = func.h(t)
    hd = diff(h, t)
    hdd = diff(hd,t)
    h = h.detach()
    hd = hd.detach()
    hdd = hdd.detach()

    h = torch.cat([h,torch.ones(len(h),1)],1)
    hd = torch.cat([hd, torch.zeros(len(h), 1)], 1)
    hdd = torch.cat([hdd, torch.zeros(len(h), 1)], 1)

    s1 = time.time()

    m1 = torch.tensor(1.)
    m2 = torch.tensor(1.)
    k1 = torch.tensor(2.)
    k2 = torch.tensor(4.)

    r2 = 1.5
    r1 = -1.5



    Mblock = get_block_m(m1, m2)
    Kblock = get_block_k(k1, k2)


    sns.axes_style(style='ticks')
    sns.set_context("paper", font_scale=2.3,
                    rc={"font.size": 30, "axes.titlesize": 25, "axes.labelsize": 20, "axes.legendsize": 20,
                        'lines.linewidth': 2.5})

    sns.set_palette('deep')
    sns.set_color_codes(palette='deep')

    losses = []
    pred_ys = []
    pred_yds = []

    for i in range(50):
        if i == 0:
            true_y0 = torch.tensor([1.,0.]).reshape(1,2)#((r2 - r1) * torch.rand(2) + r1).reshape(1, 2)
        else:
            true_y0 = ((r2 - r1) * torch.rand(2) + r1).reshape(1,2)#torch.tensor([1., 1.]).reshape(1, 2)
        true_y0dot =  torch.tensor([0.,0.]).reshape(1,2)#((r2 - r1) * torch.rand(2) + r1).reshape(1,2)#torch.tensor([1., 3.]).reshape(1, 2)

        with torch.no_grad():
            s1 = time.time()
            wout = get_wout(h, hd,hdd, true_y0,true_y0dot,m1,m2,k1,k2, t.detach())
            print(time.time()-s1)
            nwout = torch.cat([wout[:args.hidden_size+1,0].reshape(-1,1),wout[args.hidden_size+1:,0].reshape(-1,1)],1)
            pred_y = h @ nwout
            pred_yd = hd @ nwout
            pred_yddot = hdd @ nwout
            loss_diffeq = (Mblock @ pred_yddot.t()).t() + (Kblock @ pred_y.t()).t()
            pred_ys.append(pred_y)
            pred_yds.append(pred_yd)
        if i == 0:
            print(pred_ys[-1])
        losses.append((loss_diffeq**2).mean(1).detach().numpy())
    print('final loss mean')
    print(np.mean(losses),np.std(losses))

    f, (a0) = plt.subplots(1,1, figsize=(6, 6))

    for i,(pred_y,pred_yd) in enumerate(zip(pred_ys,pred_yds)):
        if i ==0:
            a0.plot(pred_y[:, 0], pred_yd[:, 0],c='g',label=r'$x_1$',linewidth=5)
            a0.plot(pred_y[:, 1], pred_yd[:, 1],c='b',label=r'$x_2$',linewidth=5)

        else:
            a0.plot(pred_y[:, 0], pred_yd[:, 0],c='black',alpha=0.02)
            a0.plot(pred_y[:, 1], pred_yd[:, 1],c='black',alpha=0.02)


        a0.set_xlabel(r'$\psi$')
        a0.set_ylabel(r'$\dot{\psi}$')

        plt.legend()
        plt.savefig('beats_ics_1.pdf',dpi=2400,bbox_inches='tight')

    f, (a1, a2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(6, 6), sharex=True)


    for i,(pred_y, pred_yd) in enumerate(zip(pred_ys, pred_yds)):
        if i == 0:
            a1.plot(np.arange(len(pred_y)) * args.dt, pred_y[:, 0], c='g',linewidth=5)
            a1.plot(np.arange(len(pred_y)) * args.dt, pred_y[:, 1], c='b',linewidth=5)
        else:
            a1.plot(np.arange(len(pred_y)) * args.dt, pred_y[:, 0], c='black', alpha=0.02)
            a1.plot(np.arange(len(pred_y)) * args.dt, pred_y[:, 1], c='black', alpha=0.02)

    a1.set_ylabel(r'$\psi$')

    losses = np.stack(losses, 1)
    a2.set_yscale('log')
    a2.plot(np.arange(len(losses)) * args.dt, losses.mean(1), c='royalblue')
    a2.set_ylabel(r'Residuals')
    a2.set_xlabel(r'Time (s)')

    plt.savefig('beats_ics_2.pdf', dpi=2400, bbox_inches='tight')
