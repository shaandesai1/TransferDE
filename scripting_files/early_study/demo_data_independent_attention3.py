import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
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

true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)


class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y ** 3, true_A)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')


def get_batch():
    s = torch.from_numpy(
        np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
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
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1],
                     'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(),
                     pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        #         ax_vecfield.cla()
        # ax_vecfield.set_title('Learned Vector Field')
        # ax_vecfield.set_xlabel('x')
        # ax_vecfield.set_ylabel('y')
        #
        # y, x = np.mgrid[-2:2:21j, -2:2:21j]
        # dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
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


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()
        self.key_length = 2
        self.query_length = 2
        self.value_length = 2
        self.z_net = nn.Sequential(
            nn.Linear(4, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100,2)
        )

        self.key_net = nn.Sequential(
            nn.Linear(2, 2)
        )
        self.query_net = nn.Sequential(
            nn.Linear(2, 2)
        )

        self.value_net = nn.Sequential(
            nn.Linear(2, 2)
        )

        # output is 1 since the context length across time should be k
        # self.key_net = nn.Sequential(
        #     nn.Linear(2, 10),
        #     nn.Tanh(),
        #     nn.Linear(10,10),
        #     nn.Tanh(),
        #     nn.Linear(10,self.key_length)
        # )
        # self.query_net = nn.Sequential(
        #     nn.Linear(2, 10),
        #     nn.Tanh(),
        #     nn.Linear(10, 10),
        #     nn.Tanh(),
        #     nn.Linear(10, self.query_length)
        # )
        # self.value_net = nn.Sequential(
        #     nn.Linear(2, 50),
        #     nn.Tanh(),
        #     nn.Linear(50,50),
        #     nn.Tanh(),
        #     nn.Linear(50, self.value_length)
        # )

    # def get_key(self,t,y):
    def forward(self, t, inputs):
        # print(inputs.shape)
        N = int(len(inputs) // 3)
        y = inputs[:N].reshape(-1, 1, 2)
        c = inputs[N:2*N].reshape(-1, 1, 2)
        A = inputs[2*N:].reshape(-1, 1, 2)

        z_hidden = self.z_net(torch.cat([y,c],-1))
        key = self.key_net(y)
        query = self.query_net(y)
        value = self.value_net(y)
        # print(y.shape)
        # print(query.squeeze(1).shape,key.squeeze(1).T.shape)

        alpha = torch.diagonal(torch.matmul(query.squeeze(1), key.squeeze(1).T), 0).reshape(-1, 1, 1)
        alpha_reshape = alpha.squeeze().repeat_interleave(2).reshape(-1, 1, 2)
        Adot = torch.exp(alpha_reshape)
        #
        if t == 0:
            alpha_ = alpha_reshape/(alpha_reshape+1.e-9)
        else:
            alpha_ = Adot/(A + 1e-12)
        # print(alpha)
        # print(alpha_reshape.shape,value.shape)
        cdot = alpha_ * value
        # Adot = torch.exp(alpha_reshape)
        # print(cdot.shape,Adot.shape)
        # print(z_hidden.shape,cdot.shape,Adot.shape)


        return torch.cat([z_hidden,cdot, Adot], 0)


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


if __name__ == '__main__':

    ii = 0

    func = ODEFunc().to(device)

    optimizer = optim.Adam(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)

    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        c_vals = torch.zeros_like(batch_y0) * 0.1
        A_vals = torch.zeros_like(batch_y0)*0.1
        main_input = torch.cat([batch_y0,c_vals, A_vals], 0)
        pred_y = odeint(func, main_input, batch_t,method='euler').to(device)
        pred_ydot = torch.stack([func(0, pred_y_) for pred_y_ in pred_y])
        N = int(len(pred_y[1]) // 3)
        pred_c = pred_y[:, :N, :]
        pred_cdot = pred_ydot[:, :N, :]
        pred_c_ = (pred_c ** 3).reshape(-1, 2)
        loss = torch.mean((pred_cdot.reshape(-1, 2) - torch.mm(pred_c_, true_A)) ** 2) #+ torch.mean((pred_y[-1][N:].ravel()-1.)**2)

        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                A_valstest = torch.zeros_like(true_y0.reshape(-1, 1, 2))*0.1
                c_valstest = torch.zeros_like(true_y0.reshape(-1,1,2)) * 0.1
                main_input_test = torch.cat([true_y0.reshape(-1, 1, 2),c_valstest, A_valstest], 0)

                pred_y = odeint(func, main_input_test, t)
                N = int(len(pred_y[1]) // 3)
                pred_c = pred_y[:, :N, :].reshape(-1, 1, 2)
                # print(true_y.shape)
                loss = torch.mean(torch.abs(pred_c - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_c, func, ii)
                ii += 1

        end = time.time()
