import argparse

def parse_args_(str_):
	parser = argparse.ArgumentParser(str_)
	parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
	parser.add_argument('--tmax', type=float, default=5.)
	parser.add_argument('--dt', type=int, default=0.1)

	parser.add_argument('--method_rc', type=str, choices=['euler'], default='euler')
	parser.add_argument('--wout', type=str, default='analytic')
	parser.add_argument('--paramg', type=str, default='lin')

	parser.add_argument('--niters', type=int, default=1000)
	parser.add_argument('--hidden_size', type=int, default=200)

	parser.add_argument('--test_freq', type=int, default=10)

	parser.add_argument('--viz', action='store_false')
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--adjoint', action='store_false')
	parser.add_argument('--n_timepoints', type=int, default=100)
	parser.add_argument('--random_sampling', action='store_false')
	parser.add_argument('--regularization', action='store_false')
	parser.add_argument('--l1', type=int, default=0)
	parser.add_argument('--ridge', action='store_true')
	return parser

# def optimize(a0 = lambda t: t**2,#-(5./t + t)#-3*t**2


#			 ):

def parse_args_bundles_(str_):
	parser = parse_args_(str_)
	parser.add_argument('--niters_test', type = int, default =15000)
	parser.add_argument('--num_bundles', type =int, default= 10)
	parser.add_argument('--l1_reg_strength', type = int, default = 0)
	parser.add_argument('--num_bundles_test', type = int, default = 100)
	#parser.add_argument('--test_freq', type = int, default = 10)
	
	parser.add_argument('--ffnn_bias', action = 'store_false')
	parser.add_argument('--plot_pca', action = 'store_true')
	parser.add_argument('--plot_tsne', type = int, default = 0)
	parser.add_argument('--no_bias_at_inference', action = 'store_true')
	parser.add_argument('--force_bias', type = int, default = 0)
	parser.add_argument('--num_forces', type = int, default = 0)
	parser.add_argument('--ic_tr_range', nargs="+", default =[-10, 10])
	parser.add_argument('--ic_te_range', nargs="+", type = int, default =[-10, 10])
	parser.add_argument('--exp_name', type = str, default = "")


	#parser.add_argument('--', action = 'store_false')
	parser.add_argument('--save', dest='save', action='store_true')
	parser.add_argument('--no-save', dest='save', action='store_false')
	parser.add_argument('--evaluate_only', dest = 'evaluate_only' ,action='store_true')
	parser.set_defaults(save=True)

	return parser
