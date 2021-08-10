"""
time variable doesn't work as well
"""

import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=200)
parser.add_argument('--batch_time', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_false')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_false')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

xinits = torch.distributions.Uniform(1.8, 2.2).sample([20, 1])

yinits = torch.ones_like(xinits) * 0.5

newics = torch.cat([xinits.reshape(-1, 1), yinits.reshape(-1, 1)], 1)
# print(newics)
true_y0 = torch.tensor([[1.5, 0.5]]).to(device)
# true_y0 = torch.cat([torch.tensor([[2., 0.5]]),newics],0)

t = torch.linspace(0., 2., args.data_size).to(device)


# true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)


class Lambda(nn.Module):

    def forward(self, t, y):
        x, p = y[:, 0], y[:, 1]
        xd = p
        pd = -x
        return torch.cat([xd.reshape(-1, 1), pd.reshape(-1, 1)], 1)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')
    true_ydot = Lambda()(t, true_y0)


# print(true_y.shape)
# print(true_ydot)
def get_batch():
    s = 0  # torch.from_numpy(
    # np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s][0].reshape(1, -1)  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i][0].reshape(1, -1) for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):
    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        for i in range(1):
            ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, i, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, i, 1],
                         'g-')
            ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, i, 0], '--', t.cpu().numpy(),
                         pred_y.cpu().numpy()[:, i, 1], 'b--')
        # ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        # ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        for i in range(1):
            ax_phase.plot(true_y.cpu().numpy()[:, i, 0], true_y.cpu().numpy()[:, i, 1], 'g-')
            ax_phase.plot(pred_y.cpu().numpy()[:, i, 0], pred_y.cpu().numpy()[:, i, 1], 'b--')
        # ax_phase.set_xlim(-2, 2)
        # ax_phase.set_ylim(-2, 2)

        # ax_vecfield.cla()
        # ax_vecfield.set_title('Learned Vector Field')
        # ax_vecfield.set_xlabel('x')
        # ax_vecfield.set_ylabel('y')
        #
        # y, x = np.mgrid[-2:2:21j, -2:2:21j]
        # dydt = odefunc(torch.tensor(0.), torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        # mag = np.sqrt(dydt[:, 0] ** 2 + dydt[:, 1] ** 2).reshape(-1, 1)
        # dydt = (dydt / mag)
        # dydt = dydt.reshape(21, 21, 2)
        #
        # ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        # ax_vecfield.set_xlim(-2, 2)
        # ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class Transformer(nn.Module):

    def __init__(self, ndims):
        super(Transformer, self).__init__()
        # self.ndims
        self.upper = nn.Linear(ndims + 1, 2, bias=False)

        torch.nn.init.xavier_uniform(self.upper.weight)

    def forward(self, inps):
        return self.upper(inps)


class ODEFunc(nn.Module):

    def __init__(self, number_dims):
        super(ODEFunc, self).__init__()
        self.number_dims = number_dims
        self.alpha = 0.8

        self.net = nn.Sequential(
            nn.Linear(self.number_dims, self.number_dims),
            nn.Tanh(),
            # nn.Linear(self.number_dims*2, self.number_dims),
            # nn.Tanh(),
            # nn.Linear(100, 100),
            # nn.Tanh(),
            # #last dim determines how deep we want z_dimension
            # nn.Linear(100,self.number_dims)
        )

        # self.upper = nn.Sequential(
        #     nn.Linear(2,2)
        # )
        # self.lower = nn.Sequential(
        #     nn.Linear(1,2)
        # )
        # for m in self.net.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, mean=0, std=0.1)
        #         nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # first = self.upper(y)
        # second = self.lower(t.reshape(-1,1))
        return self.net(y)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


NDIMZ = 40

if __name__ == '__main__':

    ii = 0
    func = ODEFunc(NDIMZ).to(device)
    func.load_state_dict(torch.load('func_dict'))
    func.eval()
    # freeze all weights
    for param in func.parameters():
        param.requires_grad = False

    transf = Transformer(NDIMZ).to(device)
    transf.load_state_dict(torch.load('transf_dict'))

    optimizer = optim.SGD(list(transf.parameters()), lr=1e-2)

    # l2_lambda = 0.01

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,2000,0.1)
    end = time.time()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    batch_y0, batch_t, batch_y = get_batch()
    zinit = torch.zeros(NDIMZ).reshape(1, NDIMZ)
    integ = odeint(func, zinit, batch_t, method='euler').to(device)
    integdot = torch.stack([func(batch_t[i], integ[i]) for i in range(len(integ))])
    bias = torch.ones((len(batch_t), 1, 1))
    integ = torch.cat([integ, bias], 2)
    bias2 = torch.zeros((len(batch_t), 1, 1))
    gt = (1. - torch.exp(-batch_t)).repeat_interleave(1 + 1).reshape(-1, 1, 1 + 1)
    gtd = (torch.exp(-batch_t)).repeat_interleave(1 + 1).reshape(-1, 1, 1 + 1)
    # print(integ)
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        # scheduler.step()
        pred_y = batch_y0.reshape(1, 1, 2) + gt * transf(integ)
        pred_ydot = gt * transf(torch.cat([integdot, bias2], 2)) + gtd * transf(integ)
        x, p = pred_y[:, :, 0], pred_y[:, :, 1]
        xd = p
        pd = -x
        sdothat = torch.cat([xd.reshape(-1, 1), pd.reshape(-1, 1)], 1)
        # print(pred_ydot.shape,sdothat.shape)
        # print(pred_y[0])

        # l2_reg = torch.tensor(0.)
        # for param in transf.parameters():
        #     l2_reg += torch.norm(param)

        loss = torch.mean(
            (pred_ydot.squeeze() - sdothat) ** 2)  # + l2_reg  # +# torch.mean((pred_y[0]-batch_y0.reshape(1,1,2))**2)
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                gt = (1. - torch.exp(-t)).repeat_interleave(1 + 1).reshape(-1, 1, 1 + 1)
                zinit = torch.zeros(NDIMZ).reshape(1, NDIMZ)
                integ = odeint(func, zinit, t, method='euler')
                bias = torch.ones((len(t), 1, 1))
                integ = torch.cat([integ, bias], 2)
                pred_y = true_y0.reshape(1, 1, 2) + gt * transf(integ)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()

    # torch.save(func.state_dict(), 'main_method')
