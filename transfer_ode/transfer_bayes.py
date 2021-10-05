
from collections import OrderedDict
import copy
from dataclasses import dataclass
import json
from math import ceil, fabs
import numpy as np
import torch
from scipy.sparse import csr_matrix
# from multiprocessing import Pool as mp_Pool
# import multiprocessing
import pylab as pl
from IPython import display
from copy import deepcopy

from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from matplotlib.ticker import MaxNLocator

import types
import functools

import itertools
#import logging
import ray
from time import sleep
import parser_args
from sklearn.preprocessing import MinMaxScaler

import base_solver_ffnn_bundles as tode_base

import math
from dataclasses import dataclass
import torch

#botorch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.optim import optimize_acqf
#from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from decimal import Decimal

#gpytorch
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior

#torch (we import functions from modules for small speed ups in performance)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch import nn, optim
from torch.autograd import Variable
from torch.autograd import grad
from torch.autograd import Function as Function
from torch.quasirandom import SobolEngine
from torch import matmul, pinverse, hstack, eye, ones, zeros, cuda, Generator, rand, randperm, no_grad, normal, tensor, vstack, cat, dot, ones_like, zeros_like
from torch import clamp, prod, where, randint, stack
from torch import device as torch_device
from torch.cuda import is_available as cuda_is_available
from torch.nn import Linear, MSELoss, Tanh, NLLLoss, Parameter

#other packages
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import time


locals().update(tode_base.__dict__)

if __name__ == "__main__":
    CUDAA = torch.cuda.is_available()
    if CUDAA:
        print("cuda is available")
        n_gpus = 0.1
    else:
        print("cuda is not available")
        n_gpus = 0
else:
    CUDAA = torch.cuda.is_available()
    if CUDAA:
        n_gpus = 0.1
    else:
        n_gpus = 0

@dataclass
class TurboState:
    """
    This is from BOTorch. The Turbo state is a stopping condition.

    #TODO finish description

    Arguments:
        dim: 
        batch_size:
        length_min:
        length_max:
        failure_counter:
        success_counter:
        success_tolerance:
        best_value:        the best value we have seen so far
        restart_triggered: has a restart been triggered? If yes BO will terminate.
    """
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10 # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = ceil(
            max([5.0 , float(self.dim) ]) #/ self.batch_size / self.batch_size
        )

def get_initial_points(dim, n_pts, device, dtype):
    sobol = SobolEngine(dimension=dim, scramble=True)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init

def update_state(state, Y_next):
    """Updates the turbo state (checks for stopping condition)
    Essentially this checks our TURBO stopping condition.
    
    Arguments:
        state:  the Turbo state
        Y_next: the most recent error return by the objective function

    """
    if max(Y_next) > state.best_value + 1e-3 * fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True

    return state


def generate_batch(
    state,
    model,
    X,  
    Y,
    batch_size,
    n_candidates=None,
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts",
    dtype = torch.float32,
    device = None
):
    """generate a batch for the Bayesian Optimization

    Arguments:
        state: the TURBO state (a stopping metric)
        model: the GP (Gaussian Process) BOTorch model
        X: points evaluated (a vector of hyper-parameter values fed to the objective function) # Evaluated points on the domain [0, 1]^d in original example, not ours.
        Y: Function values
        n_candidates: Number of candidates for Thompson sampling
        num_restarts:
        raw_samples:
        acqf: acquisition function (thompson sampling is preferred)
    """
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        drawn = False
        while not drawn:

            sobol = SobolEngine(dim, scramble=True)
            pert = sobol.draw(n_candidates, dtype=dtype).to(dtype=dtype, device=device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # Create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = (
                rand(n_candidates, dim, dtype=dtype, device=device)
                <= prob_perturb
            )
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, randint(0, dim - 1, size=(len(ind),), device=device)] = 1

            # Create candidate points from the perturbations and the mask        
            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]

            # Sample on the candidate points
            try:
                thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
                with torch.no_grad():  
                    X_next = thompson_sampling(X_cand, num_samples=batch_size)
                    drawn = True
            except:
                # thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
                # X_next = thompson_sampling(X_cand+torch.rand_like(X_cand)*0.002, num_samples=batch_size)
                # drawn = True
                # #assert False, 'failed to draw from thompson_sampling'
                # pass
                # try:
                #     ei = qExpectedImprovement(model, Y.max(), maximize=True)
                #     with torch.no_grad():  
                #         X_next, acq_value = optimize_acqf(
                #             ei,
                #             bounds=stack([tr_lb, tr_ub]),
                #             q=batch_size,
                #             num_restarts=num_restarts,
                #             raw_samples=raw_samples,
                #         )
                # except:
                X_next = torch.rand_like(X_cand[0].reshape(1,-1))

        


    elif acqf == "ei":
        ei = qExpectedImprovement(model, train_Y.max(), maximize=True)
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    return X_next



def eval_objective_remote(parallel_args_id, parameters, dtype = None, device = None, plot_type = "error",  *args, 
                            remote = True):
    """
    This version of the RC helper function

    Parameters
    -------
    parameters: torch.tensor
        a torch.tensor of the hyper-paramters drawn from the BO_step at time t

    plot_type

    Returns
    -------

    """
    parameter_lst, X_turbo_batch, trust_region_ids = parameters
        
    num_processes = len(parameter_lst)
    
    if not remote:

        results = []

        for i, params in enumerate(parameter_lst):
            result = execute_objective(parallel_args_id, params, X_turbo_batch[i], trust_region_ids[i])
            results.append(result)
    else:

        results = ray.get([execute_objective_remote.remote(parallel_args_id, params, X_turbo_batch[i], trust_region_ids[i]) for i, params in enumerate(parameter_lst)])

    scores, result_dicts = list(zip(*results)) 

    k = best_score_index = np.argmin(scores)

    X_turbo_specs = torch.vstack([result_dict["X_turbo_spec"] for result_dict in result_dicts])

    trust_region_ids = [result_dict["trust_region_id"] for result_dict in result_dicts]

    batch_dict = {"pred" : result_dicts[k]["pred"], 
                  "y" : result_dicts[k]["val_y"], 
                  "trust_region_ids": trust_region_ids, 
                  "best_score" : min(scores)}
   
    for i, score in enumerate(scores):
        if not i:
            Scores_ = [score]
        else:
            Scores_.append(score)

    Scores_ = tensor(Scores_, dtype = dtype, device = device, requires_grad = False).unsqueeze(-1)

    return X_turbo_specs, -Scores_, batch_dict


def execute_objective(parallel_arguments, parameters, X_turbo_spec, trust_region_id, args):#arguments):
    """ Function at the heart of RCTorch, train a network on multiple rounds of cross-validated train/test info, then return the average error.
    This method also deals with dispatching mutliple series to the objective function if there are multiple, and aggregates the returned scores
        by averaging.

    Parameters
    ----------
        arguments: a list of arguments that have been put into dictionary form for multiprocessing convenience
        upper_error_limit: brutal error clipping upper bound, 'nan' error function returns the maximum upper limit as well to discourage
                the algorithm from searching in that part of the parameter space.

    Returns
    -------
        tuple of score (float), the prediction and validation sets for plotting (optional), and the job id

    (we need the job id to resort and relate X and y for BO Opt which have been scrambled by multiprocessing)
    """
    args["save"] = self.save

    model, score, pred_y, true_y, filename, loss_collector, h = tode_base.optimize(args["ic_tr_range"], 
                                                                                   args["ic_te_range"], 
                                                                                   **parameters, args = args, **args)
    
    common_args = {"X_turbo_spec" : X_turbo_spec, "trust_region_id" : trust_region_id}
    return float(score), {"pred": pred_y, 
                          "val_y" : true_y, 
                          "score" : score, 
                                    **common_args}

@ray.remote(num_gpus=n_gpus, max_calls=1)
def execute_objective_remote(parallel_arguments, parameters, X_turbo_spec, trust_region_id, test = False):#arguments):
    """ Function at the heart of RCTorch, train a network on multiple rounds of cross-validated train/test info, then return the average error.
    This method also deals with dispatching mutliple series to the objective function if there are multiple, and aggregates the returned scores
        by averaging.

    Parameters
    ----------
        arguments: a list of arguments that have been put into dictionary form for multiprocessing convenience
        upper_error_limit: brutal error clipping upper bound, 'nan' error function returns the maximum upper limit as well to discourage
                the algorithm from searching in that part of the parameter space.

    Returns
    -------
        tuple of score (float), the prediction and validation sets for plotting (optional), and the job id

    (we need the job id to resort and relate X and y for BO Opt which have been scrambled by multiprocessing)
    """
    if test:
        print("hi")
    else:

        model, score, pred_y, true_y, filename, loss_collector, h = tode_base.optimize(args["ic_tr_range"], 
                                                                                       args["ic_te_range"], 
                                                                                       **parameters, args = args)
        
        common_args = {"X_turbo_spec" : X_turbo_spec, "trust_region_id" : trust_region_id}
        return float(score), {"pred": pred_y, 
                              "val_y" : true_y, 
                              "score" : score, 
                                        **common_args}


class Turbo_Bayes:
    def __init__(self, 
                 bounds,
                 x = None, y = None,
                 initial_samples=50,
                 turbo_batch_size=1, 
                 n_jobs = 8,
                 random_seed=42, 
                 feedback=None, 
                 verbose=True,
                 interactive = True,                  
                 length_min = 2**(-9), 
                 device = None, 
                 success_tolerance = 3, 
                 dtype = torch.float32,
                 track_in_grad = False,
                 dt = None,
                 n_trust_regions = None,
                 cv_samples = 1,
                 max_evals = 30,
                 store_path = None,
                 save = False

                 ):
        for key, val in locals().items():
                if key != 'self':
                    setattr(self, key, val)
        #self.random_state = Generator().manual_seed(self.random_seed + 2)

        self.batch_size = self.n_jobs

        self.parameters = OrderedDict(bounds) 
        self._errorz, self._errorz_step = [], []
        self.free_parameters = []
        self.fixed_parameters = []

        # if not self.windowsOS:
        #     try:
        #         multiprocessing.set_start_method('spawn')
        #     except:
        #         pass
        if not device:
            self.device = torch_device("cuda" if cuda_is_available() else "cpu")
        else:
            self.device = device
        if self.device == torch_device('cuda'):
            torch.cuda.empty_cache()
        

        
        print("FEEDBACK:", feedback, ", device:", device)

        

        #self.Distance_matrix = Distance_matrix

        # Normalize bounds domains and remember transformation
        self.scaled_bounds, self.bound_scalings, self.bound_intercepts = self.normalize_bounds(self.parameters)


    def normalize_bounds(self, bounds):
        """Makes sure all bounds feeded into GPyOpt are scaled to the domain [0, 1],
        to aid interpretation of convergence plots.

        Scalings are saved in instance parameters.

        Parameters
        ----------
        bounds : dicts
            Contains dicts with boundary information

        Returns
        -------
        scaled_bounds, scalings, intercepts : tuple
            Contains scaled bounds (list of dicts in GPy style), the scaling applied (numpy array)
            and an intercept (numpy array) to transform values back to their original domain
        """
        scaled_bounds = []
        scalings = []
        intercepts = []
        
        non_fixed_params = []
        
        print(self.device)
        
        for name, domain in self.bounds.items():
            # Get any fixed parmeters
            if type(domain) == int or type(domain) == float:
                # Take note
                self.fixed_parameters.append(name)

            # Free parameters
            elif type(domain) == tuple:
                # Bookkeeping
                self.free_parameters.append(name)

                # Get scaling
                lower_bound = min(domain)
                upper_bound = max(domain)
                scale = upper_bound - lower_bound

                # Transform to [0, 1] domain
                #scaled_bound = {'name': name, 'type': 'continuous', 'domain': (0., 1.)} #torch.adjustment required
                non_fixed_params.append(name)
                
                # Store
                #scaled_bounds.append(scaled_bound)
                scalings.append(scale)
                intercepts.append(lower_bound)
            else:
                raise ValueError("Domain bounds not understood")
        
        n_hyperparams = len(non_fixed_params)
        
        scaled_bounds = cat([zeros(1,n_hyperparams, device = self.device), 
                                   ones(1, n_hyperparams, device = self.device)], 0)
        return scaled_bounds, tensor(scalings, device = self.device, requires_grad = False), tensor(intercepts, device = self.device, requires_grad = False) #torch.adjustment required


    def convert_params(self, parameters):
        return [self.construct_arguments(parameters[i, :]) for i in  range(parameters.shape[0])]

    def _turbo_split_initial_samples(self, X_inits, n_jobs, turbo_id_override = None):

        """This function splits and prepares the initial samples in order to get initialization done."""
        batch_size = n_jobs
        nrow = X_inits[0].shape[0]
        n_clean_batches = nrow // batch_size
        final_batch_size = nrow-n_clean_batches*n_jobs

        initial_batches = []
        turbo_iter = []

        turbo_iter += [batch_size] * n_clean_batches

        if final_batch_size != 0:
            turbo_iter += [final_batch_size]

        for turbo_id, X_init in enumerate(X_inits):

            #if there is just one that we want to update, ie we are doing a restart:
            if turbo_id_override:
                turbo_id = turbo_id_override
            
            for i in range(n_clean_batches):
                
                if len(X_init) > batch_size:
                        X_batch = X_init[ (i*batch_size) : ((i+1)*batch_size), : ]
                        initial_batches.append((self.convert_params(X_batch), X_batch, [turbo_id] * len(X_batch)))
                else:
                    if final_batch_size == 0:
                        pass
                    else:

                        X_batch = X_init[ (nrow - final_batch_size) :, : ]
                        initial_batches.append((self.convert_params(X_batch), X_batch, [turbo_id] * len(X_batch)))
            # else:
            #     initial_batches.append((self.convert_params(X_init), X_init,  [turbo_id] *  len(X_init)))

        return initial_batches, turbo_iter

    def denormalize_bounds(self, normalized_arguments):
        """Denormalize arguments to feed into model.

        Parameters
        ----------
        normalized_arguments : numpy array
            Contains arguments in same order as bounds

        Returns
        -------
        denormalized_arguments : 1-D numpy array
            Array with denormalized arguments

        """
        denormalized_bounds = (normalized_arguments * self.bound_scalings) + self.bound_intercepts
        return denormalized_bounds

    def construct_arguments(self, x):
        """Constructs arguments for ESN input from input array.

        Does so by denormalizing and adding arguments not involved in optimization,
        like the random seed.

        Parameters
        ----------
        x : 1-D numpy array
            Array containing normalized parameter values

        Returns
        -------
        arguments : dict
            Arguments that can be fed into an ESN

        """

        # Denormalize free parameters
        denormalized_values = self.denormalize_bounds(x)
        arguments = dict(zip(self.free_parameters, denormalized_values.flatten()))
        

        self.log_vars = ['connectivity', 'llambda', 'llambda2', 'enet_strength',
                         'noise', 'regularization', 'dt', 'gamma_cyclic', 'sigma',
                         'lr', 'optimizer_loss_shift'
                         #'input_connectivity', 'feedback_connectivity'
                         ]


        # Add fixed parameters
        for name in self.fixed_parameters:
            value = self.bounds[name]
            arguments[name] = value
            # if name in self.log_vars:
            #     arguments[name] = 10. ** value
            # else:
                

        for var in self.log_vars:
            if var in arguments:
                arguments[var] = 10. ** arguments[var]  # Log scale correction

        if 'n_nodes' in arguments:
            if type(arguments['n_nodes']) in [int, float]:
                arguments['n_nodes'] = tensor(arguments['n_nodes'], dtype = torch.int32, device = self.device, requires_grad = False)  # Discretize #torch.adjustment required
            else:
                arguments['n_nodes'] = arguments['n_nodes'].type(dtype = torch.int32).to(self.device)

        if not self.feedback is None:
            arguments['feedback'] = self.feedback
        
        for argument, val_tensor in arguments.items():
            
            try:
                arguments[argument] = arguments[argument].item()
            except:
                arguments[argument] = arguments[argument]
        return arguments

    def execute_initial_parallel_batches(self):
        
        dim = len(self.free_parameters)
        for turbo_state_id in range(self.n_trust_regions):
            self.states[turbo_state_id] = state = TurboState(dim, 
                                                             length_min = self.length_min, 
                                                             batch_size=self.batch_size, 
                                                             success_tolerance = self.success_tolerance)

        X_inits = [get_initial_points(self.scaled_bounds.shape[1], self.initial_samples, device = self.device, dtype = self.dtype) for i in range(self.n_trust_regions)]

        #self.get_cv_samples()

        objective_inputs, turbo_iter = self._turbo_split_initial_samples(X_inits, self.n_jobs)

        results = []
        for objective_input in objective_inputs:
            result_i = eval_objective_remote(self.parallel_args_id, objective_input, self.dtype, self.device)
            results.append(result_i)

        self.n_evals += self.initial_samples * self.n_trust_regions

        X_nexts, Y_nexts, batch_dicts = zip(*results)
        X_nexts, Y_nexts, batch_dicts = list(X_nexts), list(Y_nexts), list(batch_dicts)

        [self.updates(scores=result[1], batch_dict = result[2]) for i, result in enumerate(results)]

        #self.update_idx_parallel(results)


        ids = [batch_dict["trust_region_ids"] for batch_dict in batch_dicts]
        idxs = []
        for id_set in ids:
            idxs += id_set
        idxs = tensor(idxs, dtype=torch.int32, device = self.device).reshape(-1, 1)

        self._idx = vstack((self._idx, idxs))

        for i, X_next in enumerate(X_nexts):
            Y_next = Y_nexts[i]
            self.update_turbo(X_next = X_next, Y_next = Y_next)
        

    def _restart_turbo_m(self):
        self._idx = torch.zeros((0, 1), dtype=torch.int32) 

    def _turbo_m(self):
        """
        #TODO description
        """
        dim = len(self.free_parameters)

        self.n_evals = 0

        # Remember the hypers for trust regions we don't sample from
        self.hypers = [{} for _ in range(self.n_trust_regions)]

        # Initialize parameters
        self._restart_turbo_m()

        self.X_turbo = torch.zeros((0, dim), device = self.device)
        self.Y_turbo = torch.zeros((0, 1), device = self.device)

        #set up dict of turbo states
        self.states = {}

        self.execute_initial_parallel_batches()

        
        n_init = self.initial_samples

        # Run until TuRBO converges

        self.RCs_per_turbo_batch = self.n_trust_regions * self.turbo_batch_size
        self.n_normal_rounds = self.RCs_per_turbo_batch // self.n_jobs
        self.job_rounds_per_turbo_batch = self.n_normal_rounds + 1
        self.last_job_round_num_RCs = self.RCs_per_turbo_batch % self.n_jobs
        self.turbo_iter = [self.n_jobs] * self.n_normal_rounds

        if self.last_job_round_num_RCs != 0:
            
            self.turbo_iter += [self.last_job_round_num_RCs]

        
        count = 0

        while self.n_evals < self.max_evals: #not self.state.restart_triggered: 
            count += 1
            print(f'count: {count}, self.n_evals {self.n_evals}')

            # Generate candidates from each TR
            #X_cand = torch.zeros((self.n_trust_regions, self.dim), device = self.device)
            #y_cand = torch.inf * torch.ones((self.n_trust_regions, self.n_cand, self.batch_size), device = self.device) 
            X_nexts = []
            for turbo_id, round_batch_size in enumerate(range(self.n_trust_regions)):
                idx = np.where(self._idx == turbo_id)[0] 

                sub_turbo_X = self.X_turbo[idx]
                sub_turbo_Y = self.Y_turbo[idx]

                # Fit a GP model
                train_Y = (sub_turbo_Y - sub_turbo_Y.mean()) / sub_turbo_Y.std()
                likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                model = SingleTaskGP(sub_turbo_X, train_Y, likelihood=likelihood)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_model(mll)

                # Create a batch
                X_next = generate_batch(
                    state=self.states[turbo_id],
                    model=model,
                    X=sub_turbo_X,
                    Y=train_Y,
                    batch_size = self.turbo_batch_size,
                    n_candidates=min(5000, max(2000, 200 * dim)),
                    num_restarts=10,
                    raw_samples=512,
                    acqf="ts",
                    device = self.device
                )
                tuple_ = X_next, turbo_id, 
                X_next_lst = X_next.split(X_next.shape[1], dim = 1)
                X_next_tuple_lst = [ (i, X_next_i, turbo_id) for i, X_next_i in enumerate(X_next_lst)]
                X_nexts += X_next_tuple_lst

            X_nexts = sorted(X_nexts, key = lambda x: x[0])
            X_nexts = [ (x[1], x[2]) for x in X_nexts]

            start = time.time()
            #self.get_cv_samples()
            self.parallel_trust_regions = True
            if self.parallel_trust_regions:

                objective_inputs = self.combine_new_turbo_batches(X_nexts, self.n_jobs, self.turbo_iter)

                results = []
                for objective_input in objective_inputs:
                    result_i = eval_objective_remote(self.parallel_args_id, objective_input, self.dtype, self.device)
                    results.append(result_i)

                X_nexts_mod, Y_nexts, updates_dicts  = zip(*results)
                X_nexts_mod, Y_nexts, updates_dicts  = list(X_nexts_mod), list(Y_nexts), updates_dicts

                trust_regions_ids_lst  = [dictt["trust_region_ids"] for dictt in updates_dicts]

                #objective_inputs = [(self.convert_params(batch[0]), batch[0], batch[1]) for i, batch in enumerate(X_init_processed_batches)]

                [self.updates(result[1], result[2]) for i, result in enumerate(results)]

                #self.update_idx_parallel(results)

                # if self.interactive:
                #     self._train_plot_update(pred_ = updates_dicts[0]["pred"], validate_y = updates_dicts[0]["y"], steps_displayed = updates_dicts[0]["pred"].shape[0])

            # else:
            #     parameters, trust_region_id = parameters
            #     Y_nexts = []
                
            #     #the algorithm is O(n) w.r.t. cv_samples.
            #     for i in range(self.n_trust_regions):

            #         #can be parallelized:
            #         X_next = X_nexts[i]
            #         objective_input = (self.convert_params(X_next), i)
            #         Y_next, updates_dict = self.eval_objective(objective_input) 
            #         self.updates(scores = Y_next, batch_dict = updates_dict)
            #         Y_nexts.append(Y_next)
            #         self._idx = torch.vstack((self._idx, i * torch.ones((self.batch_size, 1), dtype=torch.int32)))

            #     if self.interactive:
            #         self._train_plot_update(pred_ = updates_dict[0]["pred"], validate_y = updates_dict[0]["y"], steps_displayed = batch_dict["pred"].shape[0])
            X_nexts_stacked = torch.vstack(X_nexts_mod)

            Y_nexts_stacked = torch.vstack(Y_nexts)


            trust_regions_ids = list(itertools.chain.from_iterable(trust_regions_ids_lst))
            #trust_regions_ids = np.vstack(trust_regions_ids_lst).reshape(-1,).tolist()

            lst_to_sort = [ (i, tr_id) for i, tr_id in enumerate(trust_regions_ids)]

            mask, tr_ids = zip(*sorted(lst_to_sort, key = lambda x: x[1]))
            mask = np.array(mask)
            print("mask", mask)

            X_nexts_batch = X_nexts_stacked[mask,:]
            Y_nexts_batch = Y_nexts_stacked[mask,:]

            for i in range(self.n_trust_regions):

                Y_next_spec = Y_nexts_batch[mask == i, :]
                X_next_spec = X_nexts_batch[mask == i, :]
                

                self.states[i] = update_state(state=self.states[i], Y_next=Y_next_spec)

                # Append data
                self.update_turbo(X_next = X_next_spec, Y_next = Y_next_spec)

                self._idx = torch.vstack((self._idx, torch.ones_like(Y_next_spec) * i))

                assert len(self._idx) == len(self.Y_turbo)

            self.n_evals += self.turbo_batch_size * self.n_trust_regions

            pct_complete = self.n_evals//self.max_evals
            print(f'progress: {pct_complete}%')

            #check if states need to be restarted
            for i, state in self.states.items():
                if state.restart_triggered:

                    idx_i = self._idx[:, 0] == i

                    #remove points from trust region
                    self._idx[idx_i, 0] = -1

                    self._errorz[i], self._errorz_step[i], self._length_progress[i] = [], [], []
                    print(f"{self.n_evals}) TR-{i} is restarting from: : ... #TODO")

                    self.execute_initial_parallel_batch(i)

                    #X_init = get_initial_points(self.scaled_bounds.shape[1], self.initial_samples, device = self.device, dtype = self.dtype)
                    
                    assert self.states[i].restart_triggered == False
                    
                    #{fbest:.4}")

                    #self._errorz_step[i] += [min(self._errorz[i])] * self.batch_size


            # # Print current status
            # print( 
            #     f"{len(self.X_turbo)}) Best score: {max(Y_next).item():.4f},  TR length: {self.state.length:.2e}" + 
            #     f" length {self.state.length}"# Best value:.item() {state.best_value:.2e},
            # )
            
            # print( 
            #     f"TR length: {self.state.length:.2e}," +  f" min length {self.state.length_min:.2e}"
            #     # + Best value:.item() {state.best_value:.2e},
            # )
        else:
            display.clear_output()

        
        #display.clear_output(wait=True) 
        #display.display(pl.gcf())
                    
        # Save to disk if desired
        if not self.store_path is None:
            with open(store_path, 'w+') as output_file:
                json.dump(best_arguments, output_file, indent=4)
        
        best_vals = self.X_turbo[torch.argmax(self.Y_turbo)]
        
        denormed_ = self.denormalize_bounds(best_vals)
        
        try:
            denormed_ = denormalize_bounds(best_vals)
        except:
            print("FAIL")

        #best_vals = X_turbo[torch.argmax(Y_turbo)]

        #####Bad temporary code to change it back into a dictionary
        denormed_free_parameters = list(zip(self.free_parameters, denormed_))
        denormed_free_parameters = dict([ (item[0], item[1].item()) for item in denormed_free_parameters])

        best_hyper_parameters = denormed_free_parameters
        for fixed_parameter in self.fixed_parameters:
            best_hyper_parameters = {fixed_parameter : self.bounds[fixed_parameter], **best_hyper_parameters }

        #log_vars = ['connectivity', 'llambda', 'llambda2', 'noise', 'regularization', 'dt']
        for var in self.log_vars:
            if var in best_hyper_parameters:
                best_hyper_parameters[var] = 10. ** best_hyper_parameters[var] 


                
        # Return best parameters
        return best_hyper_parameters

    def execute_initial_parallel_batch(self, turbo_state_id):
        
        #get the dimensions of the free parameters
        dim = len(self.free_parameters)

        #initlalize the turbo state for this trust region
        self.states[turbo_state_id] = state = TurboState(dim, 
                                                         length_min = self.length_min, 
                                                         batch_size=self.batch_size, 
                                                         success_tolerance = self.success_tolerance)

        #get the initial randomly sampled points
        X_init = get_initial_points(self.scaled_bounds.shape[1], self.initial_samples, device = self.device, dtype = self.dtype)

        #get the training and validation sets
        self.get_cv_samples()


        objective_inputs, turbo_iter = self._turbo_split_initial_samples([X_init], self.n_jobs, turbo_id_override = turbo_state_id)

        
        results = []
        for i, objective_input in enumerate(objective_inputs):
            print(i)
            result_i = eval_objective_remote(self.parallel_args_id, objective_input, self.dtype, self.device)
            #ray.wait()
            results.append(result_i)

        self.n_evals += self.initial_samples

        X_nexts, Y_nexts, batch_dicts = zip(*results)
        X_nexts, Y_nexts, batch_dicts = list(X_nexts), list(Y_nexts), list(batch_dicts)

        [self.updates(scores=result[1], batch_dict = result[2]) for i, result in enumerate(results)]

        self.update_idx_parallel(results)

        for i, X_next in enumerate(X_nexts):
            Y_next = Y_nexts[i]
            self.update_turbo(X_next = X_next, Y_next = Y_next)

    def update_turbo(self, X_next, Y_next):
        self.X_turbo = cat((self.X_turbo, X_next), dim=0)
        self.Y_turbo = cat((self.Y_turbo, Y_next), dim=0)

    def objective_sampler(self):
        """Splits training set into train and validate sets, and computes multiple samples of the objective function.

        This method also deals with dispatching multiple series to the objective function if there are multiple,
        and aggregates the returned scores by averaging.

        Parameters
        ----------
        parameters : array
            Parametrization of the Echo State Network

        Returns
        -------
        mean_score : 2-D array
            Column vector with mean score(s), as required by GPyOpt

        """
        # Get data
        #self.parameters = parameters
        training_y = self.y
        training_x = self.x

        
        # Set viable sample range
        
        viable_start = 0
        # Get number of series
        self.n_series = training_x.shape[1]
        viable_stop = training_x.shape[0] - self.subsequence_length

        # Get sample lengths
        self.validate_length = torch.round(tensor(self.subsequence_length * self.validate_fraction, requires_grad  = False)).type(torch.int32)
        self.train_length = self.subsequence_length - self.validate_length

        ### TORCH
        start_indices = randint(low = viable_start, high = viable_stop, size = (self.cv_samples,))
        start_indices = [index_tensor.detach() for index_tensor in start_indices]
        
        if self.random_seed == None:
            random_seeds  = randint(0, 100000, size = (self.n_res,), generator = self.random_state) #device = self.device, 
        else:
            random_seeds = [self.random_seed]

        objective_inputs = self._build_unq_dict_lst(start_indices, random_seeds)

        return self._define_tr_val(objective_inputs[0])

    def get_cv_samples(self):

        cv_samples = [self.objective_sampler() for i in range(self.cv_samples)]

        fit_inputs = []
        val_inputs = []
        for i, cv_sample in enumerate(cv_samples):
            cv_sample_score = 0
            fit_inputs.append(cv_sample[0])
            val_inputs.append(cv_sample[1])

        self.parallel_arguments["fit_inputs"]= fit_inputs
        self.parallel_arguments["val_inputs"]= val_inputs

        # self.parallel_arguments["fit_inputs"] = fit
        #

        self.parallel_args_id = ray.put(self.parallel_arguments)

    def combine_new_turbo_batches(self, sorted_lst, n_jobs, turbo_iter):
        prev_index =0
        new_batches = []
        new_turbo_ids = []
        hps = []
        for i, index in enumerate(turbo_iter):
            sub_list = sorted_lst[prev_index:index+prev_index]
            X_batch_lst, turbo_ids = zip(*sub_list)
            X_batch_lst = list(X_batch_lst)
            prev_index += index

            X_batch_spec = torch.vstack(X_batch_lst)
            hps_spec = self.convert_params(X_batch_spec)

            new_turbo_ids.append(turbo_ids)
            new_batches.append(X_batch_spec)
            hps.append(hps_spec)
        return list(zip(hps, new_batches, new_turbo_ids))

    def recover_hps(self, alternative_index = None):
        if alternative_index:
            _, best_indices = self.Y_turbo.view(-1,).topk(len(self.Y_turbo))
            #I'm too lazy to change this now but best_vals will refer to the selected hps to "recover"
            best_vals = self.X_turbo[best_indices[alternative_index]]
        else:
            best_vals = self.X_turbo[torch.argmax(self.Y_turbo)]
            
        denormed_ = self.denormalize_bounds(best_vals)

        try:
            denormed_ = denormalize_bounds(best_vals)
        except:
            print("FAIL")

        #best_vals = X_turbo[torch.argmax(Y_turbo)]

        #####Bad temporary code to change it back into a dictionaryf
        denormed_free_parameters = list(zip(self.free_parameters, denormed_))
        denormed_free_parameters = dict([ (item[0], item[1].item()) for item in denormed_free_parameters])

        best_hyper_parameters = denormed_free_parameters
        for fixed_parameter in self.fixed_parameters:
            best_hyper_parameters = {fixed_parameter : self.bounds[fixed_parameter], **best_hyper_parameters }

        #log_vars = ['connectivity', 'llambda', 'llambda2', 'noise', 'regularization', 'dt']
        for var in self.log_vars:
            if var in best_hyper_parameters:
                best_hyper_parameters[var] = 10. ** best_hyper_parameters[var] 



        # Return best parameters
        return best_hyper_parameters


    def update_idx_parallel(self, results):
        #TODO: this function is retarded, rewrite.
        idxs = []

        for i, result in enumerate(results):
            num_points = result[0].shape[0] 
            idx_spec = result[2]["trust_region_ids"]#
            idxs += idx_spec
        idxs = tensor(idxs, dtype=torch.int32, device = self.device).reshape(-1, 1)
        try:
            self._idx = torch.vstack((self._idx, idxs))
        except:
            assert False, results[0][0] 

    def optimize(self ):

        self.parallel_args_id = ray.put(args)
        self.best_score_yet = None

        self._errorz, self._errorz_step, self._length_progress = {}, {}, {}
        self._errorz["all"] = []
        for i in range(self.n_trust_regions):
            self._errorz[i], self._errorz_step[i], self._length_progress[i] = [], [], []
        
        best_hyper_parameters = self._turbo_m()

        return best_hyper_parameters

    def updates(self, scores, batch_dict):

        if not self.best_score_yet:
            self.best_score_yet = batch_dict
        elif batch_dict["best_score"] < self.best_score_yet["best_score"]:
            self.best_score_yet = batch_dict
        else:
            pass   

        trust_region_ids = batch_dict["trust_region_ids"]
        
        for i, score in enumerate(scores):
            trust_region_id = trust_region_ids[i]
            
            state = self.states[trust_region_id]
            score__ = -float(score)
            # if self.log_score:
            #     score__ = 10**score__
            self._errorz[trust_region_id].append(score__)
            self._errorz["all"].append(score__)
            self._length_progress[trust_region_id].append(state.length)

            self._errorz_step[trust_region_id] += [min(self._errorz[trust_region_id])]#* len(scores) #+= [min(self._errorz[trust_region_id])] 

__name__ = "__main__"
if __name__ == "__main__":

    bounds = {"lr" : (-4, -1), 
              "hidden_size" : (100, 300),
              "spikethreshold" : 100,#(0.05, 1),
              "gamma" : 0.95, #(0.05, 1),
              "activation_number" : (0,3),
              "n_layers" : (1,4),
              "momentum_gamma" : (0.05,1),
              "optimizer_loss_shift" : (-5, -2),
              "gamma_cyclic" : 0}#(float(np.log10(0.98)), float(np.log10(0.99999)))}

    flag = 0
    # if not flag
    parser = parser_args.parse_args_bundles_('transfer demo')
    args = parser.parse_args()
    locals().update(args.__dict__)

    args = args.__dict__

    scaler = MinMaxScaler()

    for key in bounds.keys():
        try:
            del args[key]
        except:
            pass

    bo = Turbo_Bayes(bounds = bounds, n_trust_regions = 10, n_jobs = 10, max_evals = 2000)
    bo.optimize()
    try:
        opt_hps = bo.optimize()
        print(opt_hps)
    except:
        print(bo.recover_hps())


