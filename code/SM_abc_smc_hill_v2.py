#%%
""" A simple implementation of Approximate Bayesian Inference based on
Sequential Monte Carlo. """

import time
import os
from functools import partial
from random import choices, seed
from typing import Any, Dict, List, Optional, Union
from Model_fitting_functions import *
from Models import *

import numpy as np  # type: ignore
from p_tqdm import p_umap  # type: ignore
from scipy.stats import multivariate_normal, norm, uniform  # type: ignore

from RSS_Scoring import *
#Insert wildtype parameter distribution information.
path = '../data/smc_WT_new/pars_final.out'
WT_converged_params = Out_to_DF_hill(path, model_hill, mut_name= "", all = False)
param_dist = multivariate_dis(WT_converged_params)


#Define list of parameters
parlistS: List[Dict[str, Union[str, float]]] = [{
    'name': 'log_MA_s',
    'lower_limit': -2.0,
    'upper_limit': 2.0
}, {
    'name': 'log_MB_s',
    'lower_limit': -2.0,
    'upper_limit': 2.0
}, {
    'name': 'log_MC_s',
    'lower_limit': -2.0,
    'upper_limit': 2.0
}, {
    'name': 'log_MN_s',
    'lower_limit': -2.0,
    'upper_limit': 2.0
}] 

parlistR: List[Dict[str, Union[str, float]]] = [{
    'name': 'log_MA_r',
    'lower_limit': -2.0,
    'upper_limit': 2.0
}, {
    'name': 'log_MB_r',
    'lower_limit': -2.0,
    'upper_limit': 2.0
}, {
    'name': 'log_MC_r',
    'lower_limit': -2.0,
    'upper_limit': 2.0
}, {
    'name': 'log_MN_r',
    'lower_limit': -2.0,
    'upper_limit': 2.0
}] 

parlistO: List[Dict[str, Union[str, float]]] = [{
    'name': 'log_MA_o',
    'lower_limit':-2.0,
    'upper_limit':2.0
}, {
    'name': 'log_MB_o',
    'lower_limit':-2.0,
    'upper_limit':2.0
}, {
    'name': 'log_MC_o',
    'lower_limit':-2.0,
    'upper_limit':2.0
}, {
    'name': 'log_MN_o',
    'lower_limit':-2.0,
    'upper_limit':2.0
}] 

# {
#     'name': 'log_MF_o',
#     'lower_limit':-3.0,
#     'upper_limit':3.0
# }

def score_wrapper_S(log_MA_s: float, log_MB_s: float, log_MC_s: float,
                     log_MN_s: float) -> float:
    """Wrapper function two-inducer model, to be called by the optimiser."""
    #pylint: disable=too-many-arguments

    # Make a parameter dictionary, converting the log-spaced system params
    
    #Make if/elif statements for Sensor,Regulator,Output
    #Add in additional intake for calculate distance(which mutant)
    #Generate a loop for all mutants and run sequential abc given a pardict
    #make an iteration of outputs to create new smc folders for each mutant

    rndint = np.random.randint(low=0, high=1e7)
    timeseed = time.time_ns() % 2**16
    np.random.seed(rndint+timeseed)
    seed(rndint+timeseed)
    #in case negative parameter values are selected from multivariable distribution.
    neg = True 
    c = 0
    while neg == True:
        random_params = param_dist.rvs(size=1, random_state=rndint+timeseed)
        for rand in random_params:
            if 10**rand < 0:
                c += 1
        if c == 0:
            neg = False
        else:
            c = 0 
            
    par_dict = {
    "A_s":10**random_params[0],
    "B_s":10**random_params[1],
    "C_s":10**random_params[2],
    "N_s":random_params[3],
    "MA_s":10**log_MA_s,
    "MB_s":10**log_MB_s,
    "MC_s":10**log_MC_s, 
    "MN_s":10**log_MN_s, 
    "A_r":10**random_params[4],
    "B_r":10**random_params[5],
    "C_r":10**random_params[6],
    "N_r":random_params[7],
    "MA_r":10**0.0,
    "MB_r":10**0.0,
    "MC_r":10**0.0,
    "MN_r":10**0.0,
    "A_o":10**random_params[8],
    "B_o":10**random_params[9],
    "C_o":10**random_params[10],
    "C_k":10**random_params[11],
    "N_o":random_params[12],
    "F_o":10**random_params[13],
    "MA_o":10**0.0,
    "MB_o":10**0.0,
    "MC_o":10**0.0,
    "MN_o":10**0.0,
        }

    par_list = list(par_dict.values()) 
    
    # Call the actual scoring function
    return par_list

def score_wrapper_R(log_MA_r: float, log_MB_r: float, log_MC_r: float, log_MN_r: float) -> float:
    """Wrapper function two-inducer model, to be called by the optimiser."""
    rndint = np.random.randint(low=0, high=1e7)
    
    timeseed = time.time_ns() % 2**16
    np.random.seed(rndint+timeseed)
    seed(rndint+timeseed)
    #in case negative parameter values are selected from multivariable distribution.
    neg = True 
    c = 0
    while neg == True:
        random_params = param_dist.rvs(size=1, random_state=rndint+timeseed)
        for rand in random_params:
            if 10**rand <0:
                c += 1
        if c == 0:
            neg = False
        else:
            c = 0 
                
    par_dict = {
    "A_s":10**random_params[0],
    "B_s":10**random_params[1],
    "C_s":10**random_params[2],
    "N_s":random_params[3],
    "MA_s":10**0.0,
    "MB_s":10**0.0,
    "MC_s":10**0.0,
    "MN_s":10**0.0,
    "A_r":10**random_params[4],
    "B_r":10**random_params[5],
    "C_r":10**random_params[6],
    "N_r":random_params[7],
    "MA_r":10**log_MA_r,
    "MB_r":10**log_MB_r,
    "MC_r":10**log_MC_r,
    "MN_r":10**log_MN_r,
    "A_o":10**random_params[8],
    "B_o":10**random_params[9],
    "C_o":10**random_params[10],
    "C_k":10**random_params[11],
    "N_o":random_params[12],
    "F_o":10**random_params[13],
    "MA_o":10**0.0,
    "MB_o":10**0.0,
    "MC_o":10**0.0,
    "MN_o":10**0.0,
        }

    par_list = list(par_dict.values()) 
    
    # Call the actual scoring function
    return par_list

def score_wrapper_O(log_MA_o: float, log_MB_o: float, log_MC_o: float, log_MN_o: float) -> float:
    """Wrapper function two-inducer model, to be called by the optimiser."""
    rndint = np.random.randint(low=0, high=1e7)
    
    timeseed = time.time_ns() % 2**16
    np.random.seed(rndint+timeseed)
    seed(rndint+timeseed)
    #in case negative parameter values are selected from multivariable distribution.
    neg = True 
    c = 0
    while neg == True:
        random_params = param_dist.rvs(size=1, random_state=rndint+timeseed)
        for rand in random_params:
            if 10**rand <0:
                c += 1
        if c == 0:
            neg = False
        else:
            c = 0 
            
    par_dict = {
    "A_s":10**random_params[0],
    "B_s":10**random_params[1],
    "C_s":10**random_params[2],
    "N_s":random_params[3],
    "MA_s":10**0.0,
    "MB_s":10**0.0,
    "MC_s":10**0.0,
    "MN_s":10**0.0,
    "A_r":10**random_params[4],
    "B_r":10**random_params[5],
    "C_r":10**random_params[6],
    "N_r":random_params[7],
    "MA_r":10**0.0,
    "MB_r":10**0.0,
    "MC_r":10**0.0,
    "MN_r":10**0.0,
    "A_o":10**random_params[8],
    "B_o":10**random_params[9],
    "C_o":10**random_params[10],
    "C_k":10**random_params[11],
    "N_o":random_params[12],
    "F_o":10**random_params[13],
    "MA_o":10**log_MA_o,
    "MB_o":10**log_MB_o,
    "MC_o":10**log_MC_o,
    "MN_o":10**log_MN_o,
    }

    par_list = list(par_dict.values()) 
    
    # Call the actual scoring function
    return par_list

###############################################################

def make_output_folder(name: str = "smc_hill_new") -> None:
    """Make sure the output folder exists, else make it."""
    if not os.path.isdir('../data/'+ name):
        os.mkdir('../data/'+ name)

def sample_prior_S() -> List[float]:
    """ Generate one random draw of parameters from the priors of Sensor only. """
    prior = []
    for par_entry in parlistS:
        keys = par_entry.keys()
        # If limits are given, we use a uniform distribution
        if par_entry['name'].endswith("s"):
            if "lower_limit" in keys and "upper_limit" in keys:
                lower = float(par_entry["lower_limit"])
                upper = float(par_entry["upper_limit"])
                # Note that scale parameter denotes the width of the distribution!
                # docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html
                prior.append(uniform.rvs(loc=lower, scale=upper - lower))

            # If mean and stdev are given, we use a Gaussian
            elif "mean" in keys and "stdev" in keys:
                mean = float(par_entry["mean"])
                stdev = float(par_entry["stdev"])
                prior.append(norm.rvs(loc=mean, scale=stdev))

            else:
                raise KeyError("Prior unclear.")  
        else:  
            raise KeyError("incorrect parameters")
    return prior

def sample_prior_R() -> List[float]:
    """ Generate one random draw of parameters from the priors of Sensor only. """
    prior = []
    for par_entry in parlistR:
        keys = par_entry.keys()
        # If limits are given, we use a uniform distribution
        if par_entry['name'].endswith("r"):
            if "lower_limit" in keys and "upper_limit" in keys:
                lower = float(par_entry["lower_limit"])
                upper = float(par_entry["upper_limit"])
                # Note that scale parameter denotes the width of the distribution!
                # docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html
                prior.append(uniform.rvs(loc=lower, scale=upper - lower))

            # If mean and stdev are given, we use a Gaussian
            elif "mean" in keys and "stdev" in keys:
                mean = float(par_entry["mean"])
                stdev = float(par_entry["stdev"])
                prior.append(norm.rvs(loc=mean, scale=stdev))

            else:
                raise KeyError("Prior unclear.")  
        else: 
            raise KeyError("incorrect parameters")
    return prior

def sample_prior_O() -> List[float]:
    """ Generate one random draw of parameters from the priors of Sensor only. """
    prior = []
    for par_entry in parlistO:
        keys = par_entry.keys()
        # If limits are given, we use a uniform distribution
        if par_entry['name'].endswith("o"):
            if "lower_limit" in keys and "upper_limit" in keys:
                lower = float(par_entry["lower_limit"])
                upper = float(par_entry["upper_limit"])
                # Note that scale parameter denotes the width of the distribution!
                # docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html
                prior.append(uniform.rvs(loc=lower, scale=upper - lower))

            # If mean and stdev are given, we use a Gaussian
            elif "mean" in keys and "stdev" in keys:
                mean = float(par_entry["mean"])
                stdev = float(par_entry["stdev"])
                prior.append(norm.rvs(loc=mean, scale=stdev))

            else:
                raise KeyError("Prior unclear.")  
        else: 
            raise KeyError("incorrect parameters")
    return prior

def evaluate_parametrisationS(pars: List[float]) -> float:
    """ Returns how probably a given system parametrisation is based on
    the specified priors."""
    probability = 1.0
    for par, par_entry in zip(pars, parlistS):
        keys = par_entry.keys()
        if "lower_limit" in keys and "upper_limit" in keys:
            lower = float(par_entry["lower_limit"])
            upper = float(par_entry["upper_limit"])
            probability *= uniform.pdf(par, loc=lower, scale=upper - lower)
        elif "mean" in keys and "stdev" in keys:
            mean = float(par_entry["mean"])
            stdev = float(par_entry["stdev"])
            probability *= norm.pdf(par, loc=mean, scale=stdev)
        else:
            raise KeyError("Prior unclear.")
    return probability

def evaluate_parametrisationR(pars: List[float]) -> float:
    """ Returns how probably a given system parametrisation is based on
    the specified priors."""
    probability = 1.0
    for par, par_entry in zip(pars, parlistR):
        keys = par_entry.keys()
        if "lower_limit" in keys and "upper_limit" in keys:
            lower = float(par_entry["lower_limit"])
            upper = float(par_entry["upper_limit"])
            probability *= uniform.pdf(par, loc=lower, scale=upper - lower)
        elif "mean" in keys and "stdev" in keys:
            mean = float(par_entry["mean"])
            stdev = float(par_entry["stdev"])
            probability *= norm.pdf(par, loc=mean, scale=stdev)
        else:
            raise KeyError("Prior unclear.")
    return probability

def evaluate_parametrisationO(pars: List[float]) -> float:
    """ Returns how probably a given system parametrisation is based on
    the specified priors."""
    probability = 1.0
    for par, par_entry in zip(pars, parlistO):
        keys = par_entry.keys()
        if "lower_limit" in keys and "upper_limit" in keys:
            lower = float(par_entry["lower_limit"])
            upper = float(par_entry["upper_limit"])
            probability *= uniform.pdf(par, loc=lower, scale=upper - lower)
        elif "mean" in keys and "stdev" in keys:
            mean = float(par_entry["mean"])
            stdev = float(par_entry["stdev"])
            probability *= norm.pdf(par, loc=mean, scale=stdev)
        else:
            raise KeyError("Prior unclear.")
    return probability

def generate_parametrisation(name, data,
                             processcall: Any = 0,
                             prev_parametrisations=None,
                             prev_weights=None,
                             eps_dist=10000.0,
                             kernel=None,
                             ):
    """ Generate one valid parametrisation given a set of previous
    parametrisations and their corresponding weights. The proposed new
    parametrisation will fall under the threshold eps_dist."""
    # pylint: disable=unused-argument
    # processall is a dummy variable for tracking function performance and
    # for using p_tqdm. """

    # Setting random seeds for every thread/process to avoid having the same
    # random sequence in each thread, based on both thread ID and system time [ns]
    # TODO: This is extremely dirty for now, but hopefully it works...
    np.random.seed(processcall)  # type: ignore
    rndint = np.random.randint(low=0, high=1e7)
    
    timeseed = time.time_ns() % 2**16
    np.random.seed(rndint+timeseed)
    seed(rndint+timeseed)
    # TODO: Use "kernel.random_state(rndint+timeseed)" here for more elegance :)

    # On our way to the next parametriation, we here store the distances
    # of all drawn parametrisations which were deemed possible
    # under the specified priors
    evaluated_distances = []

    # Initialise distance beyond the threshold (epsilon initial)
    current_dist = eps_dist + 1

    # Find our parametrisation:
    # If we are in the first SMC step:
    if prev_parametrisations is None:
        # We want to find a parametrisatoin
        # with distance below threshold by doing:
        while current_dist > eps_dist:
            # Sample randomly from the prior
            if name.startswith("Sensor"):
                    proposed_pars = sample_prior_S()
                    if evaluate_parametrisationS(proposed_pars) > 0:
                        par_list = score_wrapper_S(*proposed_pars)
                        current_dist = RSS_Score(param_list= par_list, model_type=model_hill, data_=data, model_specs= 'model_muts')
                        evaluated_distances.append(current_dist)
            elif name.startswith("Regulator"):
                    proposed_pars = sample_prior_R()
                    if evaluate_parametrisationR(proposed_pars) > 0:
                        par_list = score_wrapper_R(*proposed_pars)
                        current_dist = RSS_Score(param_list= par_list, model_type=model_hill, data_=data, model_specs= 'model_muts')
                        evaluated_distances.append(current_dist)
            elif name.startswith("Output"):
                    proposed_pars = sample_prior_O()
                    if evaluate_parametrisationO(proposed_pars) > 0:
                        par_list = score_wrapper_O(*proposed_pars)
                        current_dist = RSS_Score(param_list= par_list, model_type=model_hill, data_=data, model_specs= 'model_muts')
                        evaluated_distances.append(current_dist)
            else:
                raise KeyError('Mutant name not recognised')
            
            

        # Once we got our parametrisation, set its weight to unity
        current_weight = 1.0

    # If we are not in the first SMC step anymore:
    else:
        # we do until we find a parametrisation below threshold:
        while current_dist > eps_dist:
            # propose a parametrisation from the previous parametrisations
            selected_pars = choices(prev_parametrisations,
                                    weights=prev_weights)[0]

            # perturb it using the constructed kernel
            # we need this re-seeding to avoid duplicates           
            timeseed = time.time_ns() % 2**16
            proposed_pars = selected_pars + kernel.rvs(random_state=rndint+timeseed)

            # Check whether the proposed parametrisation
            # is even possible under the priors
            if name.startswith("Sensor"):
                    if evaluate_parametrisationS(proposed_pars) > 0:
                        par_list = score_wrapper_S(*proposed_pars)
                        current_dist = RSS_Score(param_list= par_list, model_type=model_hill, data_=data, model_specs= 'model_muts')
                        evaluated_distances.append(current_dist)

            elif name.startswith("Regulator"):
                    if evaluate_parametrisationR(proposed_pars) > 0:
                        par_list = score_wrapper_R(*proposed_pars)
                        current_dist = RSS_Score(param_list= par_list, model_type=model_hill, data_=data, model_specs= 'model_muts')
                        evaluated_distances.append(current_dist)

            elif name.startswith("Output"):
                    if evaluate_parametrisationO(proposed_pars) > 0:
                        par_list = score_wrapper_O(*proposed_pars)
                        current_dist = RSS_Score(param_list= par_list, model_type=model_hill, data_=data, model_specs= 'model_muts')
                        evaluated_distances.append(current_dist)
            else:
                raise KeyError('Mutant name not recognised')

            # if evaluate_parametrisation(proposed_pars) > 0:
            #     # if so, store the distance/"score" of it
            #     if name.startswith("Sensor"):
            #         par_list = score_wrapper_S(*proposed_pars)
            #         score = RSS_Score(param_list= par_list, model_type=model_hill, data_=data)
            #     elif name.startswith("Regulator"):
            #         par_list = score_wrapper_R(*proposed_pars)
            #         score = RSS_Score(param_list= par_list, model_type=model_hill, data_=data)
            #     elif name.startswith("Output"):
            #         par_list = score_wrapper_O(*proposed_pars)
            #         score = RSS_Score(param_list= par_list, model_type=model_hill, data_=data)
            #     current_dist = score
            #     evaluated_distances.append(current_dist)

        # Once we got our parametrisation, the weight is calculated
        # in relation to all previously found parametrisations:
        # The weight is given by the probability of the parametrisation
        # based on the priors in the numerator.
        # This is then divided by a sum term over all previously
        # found parametrisations. The sum consists of products of
        # the weight times the probability of the distance between current
        # and previous parametrisation given the kernel.
        sum_denom = 0
        for parametrisation, weight in zip(prev_parametrisations,
                                           prev_weights):
            sum_denom += weight * kernel.pdf(proposed_pars - parametrisation)
        
        if name.startswith("Sensor"):
            current_weight = evaluate_parametrisationS(proposed_pars) / sum_denom
        elif name.startswith("Regulator"):
            current_weight = evaluate_parametrisationR(proposed_pars) / sum_denom    
        elif name.startswith("Output"):
            current_weight = evaluate_parametrisationO(proposed_pars) / sum_denom
        else:
                raise KeyError('Mutant name not recognised in weight calc')

    # Return the proposed parametrisation, its distance, its weight
    # and all the distances we encountered on our way there.
    return proposed_pars, current_dist, current_weight, evaluated_distances, par_list

def generate_parametrisations(name, data, prev_parametrisations=None,
                              prev_weights=None,
                              eps_dist: float = 10000,
                              n_pars: int = 2000,
                              kernel_factor: float = 1.0,
                              ):
    """ Call generate_parametrisation() in parallel until n_pars
    parametrisations have been accepted."""

    # If we sampled some parametrisations before, we construct a
    # multivariate-normal kernel based on their covariance matrix.
    # This is used to judge the distance of a new parametrisation from
    # the previously found parametrisations see generation_parametrisation()
    if prev_parametrisations is not None:
        previous_covar = 2.0 * kernel_factor * np.cov(
            np.array(prev_parametrisations).T)
        kernel = multivariate_normal(cov=previous_covar, allow_singular=True)
    else:
        kernel = None

    n = name
    d = data
    # The actual (parallel) call to generate_parametrisation()
    results = p_umap(
        partial(generate_parametrisation, 
                n,
                d,
                prev_parametrisations=prev_parametrisations,
                prev_weights=prev_weights,
                eps_dist=eps_dist,
                kernel=kernel), range(n_pars))
    # All the new parametrisations which have been found, ...
    new_parametrisations = [result[0] for result in results]
    # ... their corresponding distances, ...
    accepted_distances = [result[1] for result in results]
    # ... and corresponding weights.
    new_weights = [result[2] for result in results]
    # And the flattened list of lists of all evaluated distances on the way
    evaluated_distances = [res for result in results for res in result[3]]
    #Take the parameter list as well
    full_params = [result[4] for result in results]

    # Print stats of the current run
    new_weights /= np.sum(new_weights)  #Normalising weights
    acceptance_rate = n_pars / len(evaluated_distances)
    print("Acceptance rate:", acceptance_rate)
    print("Min accepted distance: ", np.min(accepted_distances))
    print("Median accepted distance: ", np.median(accepted_distances))
    print("Median evaluated distance: ", np.median(evaluated_distances))
    print("--------------------\n")

    # Return the generated set of new parametrisations from this run
    return new_parametrisations, new_weights, accepted_distances, acceptance_rate, full_params


def sequential_abc(name_, data_,
                   initial_dist: float = 1000.0,
                   final_dist: float = 0.05,
                   n_pars: int = 1000,
                   prior_label: Optional[int] = 5): #Takes name of mutant and data of mutants
    """ The main function. The sequence of acceptance thresholds starts
    with initial_dist and keeps on reducing until a final threshold
    final_dist is reached.
    prior_label can be used to restart sampling from a previous prior
    distribution in case further exploration with a lower epsilon is needed."""
    
    #initialisation
    if not os.path.isdir('../data/smc_hill_new/' + name_ + '_smc'):
        os.mkdir('../data/smc_hill_new/' + name_ + '_smc')
     #make new function
    distance = initial_dist
    not_converged = True
    last_round = False
    kernelfactor = 1.0

    if prior_label is None:
        # Start from scratch.
        pars = None
        weights = None
        iteration = 0
    else:
        # A file with the label is used to load the posterior.
        # Always use a numerical label, never 'final'
        pars = np.loadtxt(f'../data/smc_hill_new/{name_}_smc/pars_{prior_label}.out')
        weights = np.loadtxt(f'../data/smc_hill_new/{name_}_smc/weights_{prior_label}.out')
        accepted_distances = np.loadtxt(f'../data/smc_hill_new/{name_}_smc/distances_{prior_label}.out')
        distance = np.min(accepted_distances) + \
         0.95*(np.median(accepted_distances) - np.min(accepted_distances))  # type: ignore
        iteration = prior_label

    while not_converged:
        # Perform one iteration step
        iteration += 1
        print(f"SMC step {iteration} with target distance: {distance}")
        pars, weights, accepted_distances, _, all_params = generate_parametrisations(name_,data_,
            prev_parametrisations=pars,
            prev_weights=weights,
            eps_dist=distance,
            n_pars=n_pars,
            kernel_factor=kernelfactor)

        # Propose a new target distance for the subsequent step
        proposed_dist = np.min(accepted_distances) + 0.95 * (
            np.median(accepted_distances) - np.min(accepted_distances))

        # Check whether this was the final round (= we converged the last time)
        if last_round is True:
            # If so, set label accordingly, ...
            label = 'final'
            # ... break loop once we are back at the top
            not_converged = False
        else:
            # Else, the label is just the number of the iteration step.
            label = str(iteration)

        # Write results of the current step to HDD
        np.savetxt(f'../data/smc_hill_new/{name_}_smc/pars_{label}.out', pars)  # type: ignore
        np.savetxt(f'../data/smc_hill_new/{name_}_smc/weights_{label}.out', weights)  # type: ignore
        np.savetxt(f'../data/smc_hill_new/{name_}_smc/distances_{label}.out', accepted_distances)
        np.savetxt(f'../data/smc_hill_new/{name_}_smc/all_pars_{label}.out', all_params)
        # Check for convergence, defined as the proposed distance being
        # smaller than the desired final distance.
        if proposed_dist < final_dist:
            # If so, we want to perform one last iteration step
            # with the final distance
            distance = final_dist
            last_round = True
        else:
            # If not, we just continue with the proposed distance
            distance = proposed_dist  # type: ignore

    print('ABC converged succesfully!\n')
    return


# if __name__ == "__main__":
#     sequential_abc()

###########################################################

#SM code

#all mutants
#mutant_range:slice=slice(0,len(SM_names)) 

#Regulator only
mutant_range:slice=slice(11,len(SM_names)-18)

for i in SM_names[mutant_range]: 
    SM_mutant_of_interest=i
    print("Fitting Mutant:",SM_mutant_of_interest)
    start_time_per_mutant=time.time()
          
    SM_df = get_data_SM(SM_mutant_of_interest)
    


    if SM_mutant_of_interest.startswith("Sensor"):
            sequential_abc(name_=SM_mutant_of_interest, data_=SM_df)
    elif SM_mutant_of_interest.startswith("Regulator"):
            sequential_abc(name_=SM_mutant_of_interest, data_=SM_df)
    elif SM_mutant_of_interest.startswith("Output"):
            sequential_abc(name_=SM_mutant_of_interest, data_=SM_df)
    else:
                raise KeyError('Mutant name not recognised')
            
    print("\n time elapsed for this mutant \n",)
    print("--- %s seconds ---" % (time.time() - start_time_per_mutant))
    
#reg2,reg3,reg4

# if __name__ == "__main__":
#     SM_sequential_abc()
                                                                        # %%
