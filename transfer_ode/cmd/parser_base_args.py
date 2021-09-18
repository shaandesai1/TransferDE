import argparse

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