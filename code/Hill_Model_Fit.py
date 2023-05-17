#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from Models import *
from Models import model_hill
from data_wrangling import *
from itertools import chain
from itertools import repeat
from PyPDF2 import PdfMerger
import inspect
from Model_fitting_functions import *
from RSS_Scoring import *
#%%
import re
import seaborn as sns
from random import seed
import time
from Plotting_functions import *
'''Visualising Hill WT parameter distributions'''

path = '../data/smc_hill/pars_final.out'

WT_converged_params = Out_to_DF_hill(path, model_hill.model, '', all = False)

param_dist = multivariate_dis(WT_converged_params)

WT = 'Wildtype'
Paired_Density_plot(WT_converged_params, name = WT, save=True)

#%%
'''visualising whether the random selection of params from multi-v guassian is correct'''
start_time_per_mutant=time.time()
np.random.seed(0)  
rndint = np.random.randint(low=0, high=1e7)
    
timeseed = time.time_ns() % 2**16
np.random.seed(rndint+timeseed)
seed(rndint+timeseed)

WT_converged_params = Out_to_DF_hill(path, model_hill.model, '', all = False) #WT SMC_ABC results

param_dist = multivariate_dis(WT_converged_params, 13) #convert to multivariable distribution

random_params = param_dist.rvs(size=1, random_state=rndint+timeseed) #randomly selects one set of parameters

#test to see if sampling from multi-variate distribution works
param1 = []
param1 = WT_converged_params['Br'].to_numpy()

param2 = []
param2 = WT_converged_params['Cr'].to_numpy()

param1_gaus = []
param2_gaus = []
temp = param_dist.rvs(size=100, random_state=rndint+timeseed+200)

for items in temp:
    param1_gaus.append(items[5]) #Looks at the 5th and 6th parameters from the param set
    param2_gaus.append(items[6]) #Both values were taken from the same random sample

plt.scatter(param1,param2, c = 'b', label= 'WT param dist') #WT
plt.scatter(param1_gaus,param2_gaus, c ='r', label= 'Multi-variate gaus') 
#%%
'''Old strategy single parameter set visualisation'''
Hill_model = model_hill(params_list=[1]*13, I_conc=meta_dict["WT"].S)
func = Hill_model.model

params_hill_dict={"sen_params":{"A_s":10**2.881002710475187190e+00,"B_s":10**4.234114461927148021e+00,"C_s":10**2.917517255582483315e+00,"N_s":1.125732163978979461e+00},"reg_params":{"A_r":10**3.367414514911509116e+00,"B_r":10**3.841889231200538379e+00,"C_r":10**-2.898387275917982286e+00,"N_r":1.742140699208641008e+00
},"out_h_params":{},"out_params":{"A_o":10**3.039548305373660053e+00,"B_o":10**4.894183658414895888e+00,"C_o":10**-2.822117047581649274e+00,"N_o":1.688399702374762335e+00},"free_params":{"F_o":1.522445716338543864e+00}}
params_hill_list=dict_to_list(params_hill_dict)

converged_params_list_hill=Plotter(model_type=func,start_guess=params_hill_list,n_iter=1e5,method="Nelder-Mead",params_dict=params_hill_dict,custom_settings=[],tol=0.0001,mutation='Wildtype')

data = meta_dict["WT"]
RSS_Score(params_hill_list,model_hill,data, model_specs='None')

#minimisation to data using non-linear least squares
converged_params_list_hill=get_WT_params(model_type=func,start_guess=params_hill_list,n_iter=1e5,method="Nelder-Mead",params_dict=params_hill_dict,custom_settings=[],tol=0.0001) 
# %%
'''Plotting functions to visualise Hill modifier params for SM'''
#Visualising parameter distribution
from Plotting_functions import *
from RSS_Scoring import *
import time
from random import seed

def WT_score_wrapper(log_A_s: float, log_B_s: float, log_C_s: float,
                     N_s: float, log_A_r: float, log_B_r: float, log_C_r: float,
                     N_r: float, log_A_o: float, log_B_o: float, log_C_o: float,
                     N_o: float, F_o:float) -> float:
    """Intakes a list of parameters and generates a dictionary to be score"""
    #pylint: disable=too-many-arguments

    # Make a parameter dictionary, converting the log-spaced system params
    par_dict = {
        "A_s":10**log_A_s,
        "B_s":10**log_B_s,
        "C_s":10**log_C_s,
        "N_s":N_s,
        "A_r":10**log_A_r,
        "B_r":10**log_B_r,
        "C_r":10**log_C_r,
        "N_r":N_r,
        "A_o":10**log_A_o,
        "B_o":10**log_B_o,
        "C_o":10**log_C_o,
        "N_o":N_o,
        "F_o":F_o
    }
    par_list = list(par_dict.values())

    return par_list
def Visualise_SM_fit(mut_name, iter, plot_num, save:bool):
    '''Looking at the general fits to data'''
    path = f'../data/smc_SM_hill/{mut_name}_smc/all_pars_{iter}.out'  #iter = final
    path2 = f'../data/smc_SM_hill/{mut_name}_smc/pars_{iter}.out'  #only modifiers
    df = Out_to_DF_hill(path, model_hill.model_muts, mut_name, all=True)

    data=meta_dict['WT']
    SM_df = get_data_SM(mut_name)
    data_ = SM_df
    hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)
    
    fig, ((Sensor, Regulator), (Output, Stripe)) = plt.subplots(2,2, constrained_layout=True)
    Stripe.scatter(data_.S, data_.Stripe, c = 'green')
    Stripe.set_xscale('log')
    Stripe.set_yscale('log')
    Stripe.set_title(r'Full circuit with stripe')
    Output.scatter(data_.S, data_.Output, c = 'purple')
    Output.set_title(r'inducer -> S -| Output (GFP)')
    Output.set_xscale('log')
    Output.set_yscale('log')
    Regulator.scatter(data_.S, data_.Regulator, c = 'blue')
    Regulator.set_xscale('log')
    Regulator.set_yscale('log')
    Regulator.set_title(r'inducer ->S -|R (GFP output)')
    Sensor.scatter(data_.S, data_.Sensor, c = 'red')
    Sensor.set_xscale('log')
    Sensor.set_yscale('log')
    Sensor.set_title(r'inducer -> sensor (GFP output)')

    
    #Sets up a parameter set array
    np.random.seed(0)  

    set_list = []
    
    par_array = np.empty([plot_num,26])
    for i in range(1,plot_num+1):
        #selects 50 random unique parameter sets
        rndint = np.random.randint(low=0, high=1e7)
        timeseed = time.time_ns() % 2**16
        np.random.seed(rndint+timeseed)
        seed(rndint+timeseed)
        rand = np.random.randint(low=0, high=1000)
        check = set_list.count(rand)
        while check > 0:
            rand = np.random.randint(low=0, high=1000)
            check = set_list.count(rand)
            if check == 0:
                break
        set_list.append(rand)
        row_list = df.loc[rand].values.flatten().tolist()
        #convert back to normal
        par_array[i-1] = row_list

    #Records a mean of parameters or plot the lowest RSS

    # grouped = df.grouby(df.columns.mean().reset_index)
    # mean_params = list(grouped)
        
    #Keeps track of all RSS scores in a list that can be compared to the par_array    
    score_list = []
    for i in range(0,len(par_array)):
        Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model_muts(I_conc=data_.S,params_list=par_array[i])

        Stripe.plot(data_.S, Stripe_est_array, alpha = 0.1, c = 'green')
        Output.plot(data_.S, Output_est_array, alpha = 0.1, c = 'purple')
        Regulator.plot(data_.S, Regulator_est_array,alpha = 0.1, c = 'blue')
        Sensor.plot(data_.S, Sensor_est_array, alpha = 0.1, c = 'red')
        
        s = RSS_Score(par_array[i],model_hill,data_,model_specs='model_muts')
        score_list.append(round(s,3))

    # Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model_muts(I_conc=data_.S,params_list=mean_params)
    # Stripe.plot(data_.S, Stripe_est_array, alpha = 0.1, c = 'black')
    # Output.plot(data_.S, Output_est_array, alpha = 0.1, c = 'black')
    # Regulator.plot(data_.S, Regulator_est_array,alpha = 0.1, c = 'black')
    # Sensor.plot(data_.S, Sensor_est_array, alpha = 0.1, c = 'black')
    # return fig, score_list, par_array

    path = '../data/smc_hill/pars_final.out'
    WT_converged_params = Out_to_DF_hill(path, model_hill, mut_name= "", all = False)
    WT_dist = multivariate_dis(WT_converged_params)
    WT_params = list(WT_dist.rvs(size = 1, random_state=rndint+timeseed))
    WT_params = WT_score_wrapper(*WT_params)

    #plot wildtype for comparison, its random tho so how to represent it best
    # Sensor_array,Regulator_array,Output_array, Stripe_array = hill.model(I_conc=data_.S,params_list=WT_params)

    # Stripe.plot(data.S, Stripe_array, alpha = 0.7, c = 'black')
    # Output.plot(data.S, Output_array, alpha = 0.7, c = 'black')
    # Regulator.plot(data.S, Regulator_array,alpha = 0.7, c = 'black')
    # Sensor.plot(data.S, Sensor_array, alpha = 0.7, c = 'black')



#need to figure out whats wrong with model muts and why its producing dog shit
    fig.suptitle(f'{mut_name} Mutant Fitting with {plot_num} parameter sets')
    txt = f'Param set-list from all_pars_{iter}: \n'
    txt+=str(set_list)
    txt2 = f'Corresponding RSS: \n'
    txt2+=str(score_list)
    fig.text(0,-.1,txt,wrap=True, fontsize=6)
    fig.text(0,-.2,txt2,wrap=True, fontsize=6)
    if save == True:
        plt.savefig(f'../results/{mut_name}_SM_fit.pdf', format="pdf", bbox_inches="tight")

    return fig, score_list, set_list

def Visualise_SM_par_dis(mut_name, iter, saves:bool):
    path2 = f'../data/smc_SM_hill/{mut_name}_smc/pars_{iter}.out'  #only modifiers
    df2 = Out_to_DF_hill(path2, model_hill.model_muts, mut_name, all=False)
    
    return Paired_Density_plot_mut(df2, name = mut_name, save = saves)

#put a mutant name and the final or last SMC step
mut = 'Regulator2'
num = 'final'
fig, scores, set_list = Visualise_SM_fit(mut_name=mut, iter = num, plot_num = 50, save = True)

Visualise_SM_par_dis(mut_name= mut, iter = num, saves = True)
# %%

'''Calculating Epistasis Hat'''

#Currently, get_Eps takes in one set of parameters and calculates epistasis at low, medium and high inducer concentrations. Could change get_params code - get params takes a dataframe of single mutant and wildtype parameters, so I could potentially make that a thing.

#with Eps_to_Excel, all the visualisation of previous plots can be recycled to examine the distribution of epistasis values, epistasis for

#generate 1e6 random parameters from each mutant and add low medium and high fluo to lists, append to dataframe with mutant combo

#for each mutant combination for pairwise, generate 3 million epistasis values
#dataframe = {genotype:{}, low_eps:{}, med_eps:{}, high_eps:{}}
#sub_dataframe containing only genotype of interest and plot violin plots of distribution of epistasis, 1 million epistasis of wildtype.

def New_get_Eps():
    DM_df = meta_dict['DM']
    DM_names = DM_df['genotype'].unique()

#%%
from Plotting_functions import *
'''Examining the WT distribution of parameter sets between mutants'''

def get_combo_WT_df(mutants:list):
    mutant1 = mutants[0]
    mutant2 = mutants[1]

    path = f'../data/smc_SM_hill/{mutant1}_smc/all_pars_final.out' 
    df1 = Out_to_DF_hill(path, model_hill.model_muts, mutant1, all=True)
    WT1_df = df1[['As','Bs','Cs','Ns','Ar','Br','Cr','Nr','Ao','Bo','Co','No','Fo']]
    WT1_df = np.log10(WT1_df) #convert to log10
    mod_path = f'../data/smc_SM_hill/{mutant1}_smc/pars_final.out' 
    M1_mods_df = Out_to_DF_hill(mod_path, model_hill.model_muts, mutant1, all=False)

    M1_mods_df.reset_index(drop=True, inplace=True)
    WT1_df.reset_index(drop=True, inplace=True)
    M1_df = pd.concat([M1_mods_df,WT1_df], axis=1)

        
            #mutant2 modifiers
    path2 = f'../data/smc_SM_hill/{mutant2}_smc/all_pars_final.out'  
    df2 = Out_to_DF_hill(path2, model_hill.model_muts, mutant2, all=True)
    WT2_df = df2[['As','Bs','Cs','Ns','Ar','Br','Cr','Nr','Ao','Bo','Co','No','Fo']]
    WT2_df = np.log10(WT2_df) #convert to log10
    mod_path2 = f'../data/smc_SM_hill/{mutant2}_smc/pars_final.out' 
    M2_mods_df = Out_to_DF_hill(mod_path2, model_hill.model_muts, mutant2, all=False)
    #a df with modifier params after wildtype

    M2_mods_df.reset_index(drop=True, inplace=True)
    WT2_df.reset_index(drop=True, inplace=True)
    M2_df = pd.concat([M2_mods_df, WT2_df,], axis=1)

    Combined_WT = pd.concat([WT1_df,WT2_df], axis=0)


    return Combined_WT, M1_mods_df, M2_mods_df, M1_df, M2_df #all dfs are in log


# WT1_df = WT1_df.assign(Genotype= f'{mutant1}')
# WT2_df = WT2_df.assign(Genotype= f'{mutant2}')
#Paired_Density_plot_compare(Combined_WT,n,huw = 'Genotype', save=False)




# %%

'''Code to create a viable set of combined single mutant parameters'''

def get_combo_params(mutants:list):

    #Step1: obtain mu of WT, mut1, mut2 and Covariance matrix of combined wildtypes

    Combined_WT, M1_mods_df, M2_mods_df, M1_df, M2_df = get_combo_WT_df(mutants) #all log

    names = Combined_WT.keys()
    params = len(Combined_WT.columns)
    WT_matrix = np.empty(shape=(params,2000), dtype=float)
    i = 0
    for name in names:
        WT_matrix[i] = Combined_WT[name].to_numpy()
        i = i+1
    
    #range of parameters as x, means calculated
    WT_mean_list = []
    j = 0

    for m in WT_matrix:
        means = sum(m)
        means = means/len(m)
        WT_mean_list.append(means)
        j = j+1



    #generate cov matrix
    Combined_WT = Combined_WT.T
    WT_cov_matrix = np.cov(Combined_WT.values)
    #generate multivariate normal distribution
    WT_multi_norm_dis = multivariate_normal(
                        mean = WT_mean_list,
                        cov = WT_cov_matrix,
                        allow_singular = True)
    
    #Now sample a random parameter set from combined multivariate dist
    rndint = np.random.randint(low=0, high=1e7)
    
    timeseed = time.time_ns() % 2**16
    np.random.seed(rndint+timeseed)
    seed(rndint+timeseed)
    
    WT_sample = WT_multi_norm_dis.rvs(size=1, random_state=rndint+timeseed)

    #Calculate mean for mut1 (just modifiers)
    names = M1_mods_df.keys()
    params = len(M1_mods_df.columns)
    M1_mods_matrix = np.empty(shape=(params,1000), dtype=float)
    i = 0
    for name in names:
        M1_mods_matrix[i] = M1_mods_df[name].to_numpy()
        i = i+1
    
    M1_mean_list = []
    j = 0

    for m in M1_mods_matrix:
        means = sum(m)
        means = means/len(m)
        M1_mean_list.append(means)
        j = j+1

    #Generate covariance matrix of full mutant params
    names = M1_df.keys()
    params = len(M1_df.columns)
    M1_matrix = np.empty(shape=(params,1000), dtype=float)
    i = 0
    for name in names:
        M1_matrix[i] = M1_df[name].to_numpy()
        i = i+1
    M1_df = M1_df.T
    M1_cov_matrix = np.cov(M1_df.values, bias = True)
    mu1 = M1_mean_list
    mu2 = WT_mean_list
    C11 = M1_cov_matrix[0:4,0:4]
    C12 = M1_cov_matrix[0:4:,4:]
    C21 = M1_cov_matrix[4:,0:4]
    C22 = M1_cov_matrix[4:,4:]
    C22inv = np.linalg.inv(C22)
    a_minus_mu = (WT_sample - mu2)
    a_minus_mu[:, np.newaxis]
    C12C22inv = np.dot(C12,C22inv.T) #not sure if transpose is correct
    temp = np.dot(C12C22inv, a_minus_mu[:, np.newaxis])
    conditional_mu = [x+y for x, y in zip(mu1,temp.flatten().tolist())]

    conditional_cov = C11 - np.dot(C12C22inv, C21)

    M1_multi_dis = multivariate_normal(mean = conditional_mu,
                                        cov = conditional_cov, 
                                        allow_singular = True
                                                 )
    
    M1_cond_params = M1_multi_dis.rvs(size = 10, random_state=rndint+ timeseed)
    ###############################################
    #Calculate mean for mut2 (just modifiers)
    names = M2_mods_df.keys()
    params = len(M2_mods_df.columns)
    M2_mods_matrix = np.empty(shape=(params,1000), dtype=float)
    i = 0
    for name in names:
        M2_mods_matrix[i] = M2_mods_df[name].to_numpy()
        i = i+1
    
    M2_mean_list = []
    j = 0

    for m in M2_mods_matrix:
        means = sum(m)
        means = means/len(m)
        M2_mean_list.append(means)
        j = j+1

    #Generate covariance matrix of full mutant params
    names = M2_df.keys()
    params = len(M2_df.columns)
    M2_matrix = np.empty(shape=(params,1000), dtype=float)
    i = 0
    for name in names:
        M2_matrix[i] = M2_df[name].to_numpy()
        i = i+1
    M2_df = M2_df.T
    M2_cov_matrix = np.cov(M2_df.values, bias = True)
    mu1 = M2_mean_list
    mu2 = WT_mean_list
    C11 = M2_cov_matrix[0:4,0:4]
    C12 = M2_cov_matrix[0:4:,4:]
    C21 = M2_cov_matrix[4:,0:4]
    C22 = M2_cov_matrix[4:,4:]
    C22inv = np.linalg.inv(C22)
    a_minus_mu = (WT_sample - mu2)
    a_minus_mu[:, np.newaxis]
    C12C22inv = np.dot(C12,C22inv.T) 
    temp = np.dot(C12C22inv, a_minus_mu[:, np.newaxis])
    conditional_mu = [x+y for x, y in zip(mu1,temp.flatten().tolist())]

    conditional_cov = C11 - np.dot(C12C22inv, C21)

    M2_multi_dis = multivariate_normal(mean = conditional_mu,
                                       cov = conditional_cov,
                                       allow_singular = True
                                                 )
    
    M2_cond_params = M2_multi_dis.rvs(size = 10, random_state=rndint+ timeseed)

    return WT_sample, M1_cond_params, M2_cond_params


mutants = ['Regulator9','Output3']
WT, M1, M2 = get_combo_params(mutants)
# %%
# %%
'''Visualisation of pairwise/triplet fits using combined modifiers'''
import time
def Visualise_combo_mut_fit(mutants:list):
    '''Takes 2 or 3 mutants in a list and plots them to the data, enter mutants as follows [Output/Sensor/Regulator[1-10]]'''
    if len(mutants) == 2:
        mut1 = mutants[0]
        mut2 = mutants[1]

        rndint = np.random.randint(low=0, high=1e7)
        timeseed = time.time_ns() % 2**16
        np.random.seed(rndint+timeseed)
        seed(rndint+timeseed)

        size = 100
        
        #WT params
        # path = '../data/smc_hill/pars_final.out'
        # WT_converged_params = Out_to_DF_hill(path, model_hill, mut_name= "", all = False)
        # data = meta_dict['WT']
        # param_dist = multivariate_dis(WT_converged_params)
        # WT_pars_array = param_dist.rvs(size=size, random_state=rndint+timeseed)

        # #mutant1 modifiers
        # path = f'../data/smc_SM_hill/{mut1}_smc/pars_final.out'  #only modifiers
        # df1 = Out_to_DF_hill(path, model_hill.model_muts, mut1, all=False)
        # MD1 = multivariate_dis(df1)
        # mut1_pars_array = MD1.rvs(size=size, random_state=rndint+timeseed)

        # #mutant2 modifiers
        # path2 = f'../data/smc_SM_hill/{mut2}_smc/pars_final.out'  #change final if needed
        # df2 = Out_to_DF_hill(path2, model_hill.model_muts, mut2, all=False)
        # MD2 = multivariate_dis(df2)
        # mut2_pars_array = MD2.rvs(size=size, random_state=rndint+timeseed)


        #plot pairwise fit
        #selects mutant shortcode and assembles into correct mutant ID
        for i in range(0,10):
            if mut1.endswith(f'{i}'):
                if i == 0:
                    pair1 = f'{mut1[0]}1{i}'
                else:
                    pair1 = f'{mut1[0]}{i}'

            if mut2.endswith(f'{i}'):
                if i == 0:
                    pair2 = f'{mut2[0]}1{i}'
                else:
                    pair2 = f'{mut2[0]}{i}'

        DM_df = meta_dict['DM']

        #All mutants are in order of R_S_O

        if pair1.startswith('R') | pair1.startswith('S'):

            pair_mut_dict = DM_df.loc[DM_df['genotype'].str.contains(pair1)]
            if pair1.endswith('1'):
                pair_mut_dict = pair_mut_dict.loc[DM_df['genotype'].str.contains('1_')]
        elif pair1.startswith('O'):
            pair_mut_dict = DM_df.loc[DM_df['genotype'].str.contains(pair1)]

            if pair1.endswith('1'):
                pair_mut_dict = pair_mut_dict.loc[pair_mut_dict['genotype'].str.endswith('1')]
                #incase pair ends with a 1 and 10 is included
        
        if pair2.startswith('R') | pair2.startswith('S'):

            pair_mut_dict = pair_mut_dict.loc[pair_mut_dict['genotype'].str.contains(pair2)]
            if pair2.endswith('1'):
                pair_mut_dict = pair_mut_dict.loc[pair_mut_dict['genotype'].str.contains('1_')]
        elif pair2.startswith('O'):
            pair_mut_dict = pair_mut_dict.loc[pair_mut_dict['genotype'].str.contains(pair2)]

            if pair2.endswith('1'):
                pair_mut_dict = pair_mut_dict.loc[pair_mut_dict['genotype'].str.endswith('1')]
        

        pairwise_fluo = []
        for fluo in pair_mut_dict['obs_fluo_mean']:
            pairwise_fluo.append(fluo)


        pairwise_inducer = [0.00001, 0.0002, 0.2]

        ind = pd.DataFrame(pairwise_inducer)


        # plt.plot(pairwise_inducer,pairwise_fluo)
        # plt.scatter(pairwise_inducer,pairwise_fluo)
        # plt.xscale('log')
        # plt.yscale('log')

        #plot mutant fits
        hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)

        WT_pars_array = np.empty(shape=(size,13))
        Mut1_pars_array = np.empty(shape=(size,4))
        Mut2_pars_array = np.empty(shape=(size,4))

        low = []
        med = []
        high = []

        for i in range(0,size):
            WT_pars, Mut1_pars_array, Mut2_pars_array = get_combo_params(mutants)
            for Mut1_pars,Mut2_pars in zip(Mut1_pars_array,Mut2_pars_array):
            #identification of mutant types
                if mut1.startswith('Sensor') & mut2.startswith('Output'):
                    M = {'As':Mut1_pars[0],'Bs':Mut1_pars[1],'Cs':Mut1_pars[2],'Ns':Mut1_pars[3],'Ar':0.0,'Br':0.0,'Cr':0.0,'Nr':0.0,'Ao':Mut2_pars[0],'Bo':Mut2_pars[1],'Co':Mut2_pars[2],'No':Mut2_pars[3],'Fo':0.0}
                elif mut1.startswith('Sensor') & mut2.startswith('Regulator'):
                    M = {'As':Mut1_pars[0],'Bs':Mut1_pars[1],'Cs':Mut1_pars[2],'Ns':Mut1_pars[3],'Ar':Mut2_pars[0],'Br':Mut2_pars[1],'Cr':Mut2_pars[2],'Nr':Mut2_pars[3],'Ao':0.0,'Bo':0.0,'Co':0.0,'No':0.0,'Fo':0.0}
                elif mut1.startswith('Regulator') & mut2.startswith('Output'):
                    M = {'As':0.0,'Bs':0.0,'Cs':0.0,'Ns':0.0,'Ar':Mut1_pars[0],'Br':Mut1_pars[1],'Cr':Mut1_pars[2],'Nr':Mut1_pars[3],'Ao':Mut2_pars[0],'Bo':Mut2_pars[1],'Co':Mut2_pars[2],'No':Mut2_pars[3],'Fo':0.0}
                elif mut1.startswith('Regulator') & mut2.startswith('Sensor'):
                    M = {'As':Mut2_pars[0],'Bs':Mut2_pars[1],'Cs':Mut2_pars[2],'Ns':Mut2_pars[3],'Ar':Mut1_pars[1],'Br':Mut1_pars[1],'Cr':Mut1_pars[1],'Nr':Mut1_pars[1],'Ao':0.0,'Bo':0.0,'Co':0.0,'No':0.0,'Fo':0.0}
                elif mut1.startswith('Output') & mut2.startswith('Regulator'):
                    M = {'As':0.0,'Bs':0.0,'Cs':0.0,'Ns':0.0,'Ar':Mut2_pars[0],'Br':Mut2_pars[1],'Cr':Mut2_pars[2],'Nr':Mut2_pars[3],'Ao':Mut1_pars[0],'Bo':Mut1_pars[1],'Co':Mut1_pars[2],'No':Mut1_pars[3],'Fo':0.0}
                elif mut1.startswith('Output') & mut2.startswith('Sensor'):
                    M = {'As':Mut2_pars[0],'Bs':Mut2_pars[1],'Cs':Mut2_pars[2],'Ns':Mut2_pars[3],'Ar':0.0,'Br':0.0,'Cr':0.0,'Nr':0.0,'Ao':Mut1_pars[0],'Bo':Mut1_pars[1],'Co':Mut1_pars[2],'No':Mut1_pars[3],'Fo':0.0}
                else:
                    raise KeyError('Mutant names invalid 212')
            
                par_dict = {
                    "A_s":10**WT_pars[0],
                    "B_s":10**WT_pars[1],
                    "C_s":10**WT_pars[2],
                    "N_s":10**WT_pars[3],
                    "MA_s":10**M['As'],
                    "MB_s":10**M['Bs'],
                    "MC_s":10**M['Cs'],
                    "MN_s":10**M['Ns'], 
                    "A_r":10**WT_pars[4],
                    "B_r":10**WT_pars[5],
                    "C_r":10**WT_pars[6],
                    "N_r":10**WT_pars[7],
                    "MA_r":10**M['Ar'],
                    "MB_r":10**M['Br'],
                    "MC_r":10**M['Cr'],
                    "MN_r":10**M['Nr'],
                    "A_o":10**WT_pars[8],
                    "B_o":10**WT_pars[9],
                    "C_o":10**WT_pars[10],
                    "N_o":10**WT_pars[11],
                    "F_o":10**WT_pars[12],
                    "MA_o":10**M['Ao'],
                    "MB_o":10**M['Bo'],
                    "MC_o":10**M['Co'],
                    "MN_o":10**M['No'],
                    "MF_o":10**M['Fo'],
                        }
            
                par_list = list(par_dict.values())

                Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model_muts(I_conc= ind,params_list=par_list)

                low.append(Stripe_est_array.iloc[0,0])
                med.append(Stripe_est_array.iloc[1,0])
                high.append(Stripe_est_array.iloc[2,0]) 

                # plt.plot(ind, Stripe_est_array, c = 'green', alpha = 0.1)
                # plt.scatter(ind, Stripe_est_array, c = 'green', alpha = 0.1)

                # WT_pars_array[i] = WT_pars
                # Mut1_pars_array[i] = Mut1_pars
                # Mut2_pars_array[i] = Mut2_pars 

        # WT_median_store = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
        # WT_median = []
        # for object in WT_pars_array:
        #     for i in range(0,13):
        #         WT_median_store[i].append(object[i])
        # for i in range(0,13):
        #     WT_median.append(np.median(WT_median_store[i]))

        # M1_median_store = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
        # M1_median = []
        # for object in Mut1_pars_array:
        #     for i in range(0,4):
        #         M1_median_store[i].append(object[i])
        # for i in range(0,4):
        #     M1_median.append(np.median(M1_median_store[i]))

        # M2_median_store = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
        # M2_median = []
        # for object in Mut2_pars_array:
        #     for i in range(0,4):
        #         M2_median_store[i].append(object[i])
        # for i in range(0,4):
        #     M2_median.append(np.median(M2_median_store[i]))

        # WT_pars = WT_median
        # Mut1_pars = M1_median
        # Mut2_pars = M2_median
        # if mut1.startswith('Sensor') & mut2.startswith('Output'):
        #         M = {'As':Mut1_pars[0],'Bs':Mut1_pars[1],'Cs':Mut1_pars[2],'Ns':Mut1_pars[3],'Ar':0.0,'Br':0.0,'Cr':0.0,'Nr':0.0,'Ao':Mut2_pars[0],'Bo':Mut2_pars[1],'Co':Mut2_pars[2],'No':Mut2_pars[3],'Fo':0.0}
        # elif mut1.startswith('Sensor') & mut2.startswith('Regulator'):
        #     M = {'As':Mut1_pars[0],'Bs':Mut1_pars[1],'Cs':Mut1_pars[2],'Ns':Mut1_pars[3],'Ar':Mut2_pars[0],'Br':Mut2_pars[1],'Cr':Mut2_pars[2],'Nr':Mut2_pars[3],'Ao':0.0,'Bo':0.0,'Co':0.0,'No':0.0,'Fo':0.0}
        # elif mut1.startswith('Regulator') & mut2.startswith('Output'):
        #     M = {'As':0.0,'Bs':0.0,'Cs':0.0,'Ns':0.0,'Ar':Mut1_pars[0],'Br':Mut1_pars[1],'Cr':Mut1_pars[2],'Nr':Mut1_pars[3],'Ao':Mut2_pars[0],'Bo':Mut2_pars[1],'Co':Mut2_pars[2],'No':Mut2_pars[3],'Fo':0.0}
        # elif mut1.startswith('Regulator') & mut2.startswith('Sensor'):
        #     M = {'As':Mut2_pars[0],'Bs':Mut2_pars[1],'Cs':Mut2_pars[2],'Ns':Mut2_pars[3],'Ar':Mut1_pars[1],'Br':Mut1_pars[1],'Cr':Mut1_pars[1],'Nr':Mut1_pars[1],'Ao':0.0,'Bo':0.0,'Co':0.0,'No':0.0,'Fo':0.0}
        # elif mut1.startswith('Output') & mut2.startswith('Regulator'):
        #     M = {'As':0.0,'Bs':0.0,'Cs':0.0,'Ns':0.0,'Ar':Mut2_pars[0],'Br':Mut2_pars[1],'Cr':Mut2_pars[2],'Nr':Mut2_pars[3],'Ao':Mut1_pars[0],'Bo':Mut1_pars[1],'Co':Mut1_pars[2],'No':Mut1_pars[3],'Fo':0.0}
        # elif mut1.startswith('Output') & mut2.startswith('Sensor'):
        #     M = {'As':Mut2_pars[0],'Bs':Mut2_pars[1],'Cs':Mut2_pars[2],'Ns':Mut2_pars[3],'Ar':0.0,'Br':0.0,'Cr':0.0,'Nr':0.0,'Ao':Mut1_pars[0],'Bo':Mut1_pars[1],'Co':Mut1_pars[2],'No':Mut1_pars[3],'Fo':0.0}
        # else:
        #     raise KeyError('Mutant names invalid 212')

        # par_dict = {
        #         "A_s":10**WT_pars[0],
        #         "B_s":10**WT_pars[1],
        #         "C_s":10**WT_pars[2],
        #         "N_s":10**WT_pars[3],
        #         "MA_s":10**M['As'],
        #         "MB_s":10**M['Bs'],
        #         "MC_s":10**M['Cs'],
        #         "MN_s":10**M['Ns'], 
        #         "A_r":10**WT_pars[4],
        #         "B_r":10**WT_pars[5],
        #         "C_r":10**WT_pars[6],
        #         "N_r":10**WT_pars[7],
        #         "MA_r":10**M['Ar'],
        #         "MB_r":10**M['Br'],
        #         "MC_r":10**M['Cr'],
        #         "MN_r":10**M['Nr'],
        #         "A_o":10**WT_pars[8],
        #         "B_o":10**WT_pars[9],
        #         "C_o":10**WT_pars[10],
        #         "N_o":10**WT_pars[11],
        #         "F_o":10**WT_pars[12],
        #         "MA_o":10**M['Ao'],
        #         "MB_o":10**M['Bo'],
        #         "MC_o":10**M['Co'],
        #         "MN_o":10**M['No'],
        #         "MF_o":10**M['Fo'],
        #             }
            
        # par_list = list(par_dict.values())

        # Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model_muts(I_conc= ind,params_list=par_list)

        # plt.plot(ind, Stripe_est_array, c = 'r', alpha = 1.0, label='Median')
        # plt.scatter(ind, Stripe_est_array, c = 'r', alpha = 1.0)

        data = {'low':np.log(low), 'medium':np.log(med), 'high':np.log(high)}
        fluo_df = pd.DataFrame(data)
        fig, axes = plt.subplots(figsize=(10,6))

        axes2 = axes.twinx()
        point = []
        SD = []
        for obs_m, obs_sd in zip(pair_mut_dict['obs_fluo_mean'],pair_mut_dict['obs_SD']):
            point.append(obs_m)
            SD.append(obs_sd)

        point = np.log(point)
        SD = np.log(SD)
        sns.violinplot(data=fluo_df, ax=axes, orient='v', color = 'mistyrose' )
        sns.pointplot(x=np.arange(len(point)), y=point, ax=axes2, color = 'darkcyan')
        axes2.set_ylim(axes.get_ylim())

        data = meta_dict['WT']
        data_ind = [0.000012,0.000195,0.1]
        data_stripe = [data.Stripe[1],data.Stripe[5],data.Stripe[14],]
        data_stripe = np.log(data_stripe)
        sns.pointplot(x=np.arange(len(data_stripe)), y=data_stripe, ax=axes2, color = 'indigo')
        # Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model_muts(I_conc= ind,params_list=par_list)

        # plt.plot(data_ind, data_stripe, c = 'black', alpha = 1.0, label = 'WT')
        # plt.scatter(data_ind, data_stripe, c = 'black', alpha = 1.0)
        
        
        
        Rand = mpatches.Patch(color= 'mistyrose', label='Estimated fluorescence')
        Wildtype = mpatches.Patch(color= 'indigo', label='Wildtype') #Could potenitally plot the actual wildtype data
        data_set = mpatches.Patch(color= 'darkcyan', label='Pairwise data')
        plt.legend(handles=[data_set,Rand,Wildtype], bbox_to_anchor=(1, 1), title = "Legend")
        plt.title(f'Pairwise mutant fit: {pair1}_{pair2}')
        axes.set_xlabel('Inducer Concetration')
        axes.set_ylabel('Log_Fluorescence')
        axes.set_xticks(ticks=range(len(fluo_df.columns)), labels=fluo_df.columns)
    
   

mutants = ['Output3', 'Sensor9']
Visualise_combo_mut_fit(mutants)
# %%
 # elif len(mutants) == 3:
    #     mut1 = mutants[0]
    #     mut2 = mutants[1]
    #     mut3 = mutants[2]

    #     rndint = np.random.randint(low=0, high=1e7)
    #     timeseed = time.time_ns() % 2**16
    #     np.random.seed(rndint+timeseed)
    #     seed(rndint+timeseed)

    #     #WT params
    #     path = '../data/smc_hill/pars_final.out'
    #     WT_converged_params = Out_to_DF_hill(path, model_hill, mut_name= "", all = False)
    #     data = meta_dict['WT']
    #     param_dist = multivariate_dis(WT_converged_params)
    #     WT_pars_array = param_dist.rvs(size=50, random_state=rndint+timeseed)

    #     #mutant1 modifiers
    #     path = f'../data/smc_SM_hill/{mut1}_smc/pars_final.out'  #only modifiers
    #     df1 = Out_to_DF_hill(path, model_hill.model_muts, mut1, all=False)
    #     MD1 = multivariate_dis(df1)
    #     mut1_pars_array = MD1.rvs(size=50, random_state=rndint+timeseed)

    #     #mutant2 modifiers
    #     path2 = f'../data/smc_SM_hill/{mut2}_smc/pars_final.out'  #change final if needed
    #     df2 = Out_to_DF_hill(path2, model_hill.model_muts, mut2, all=False)
    #     MD2 = multivariate_dis(df2)
    #     mut2_pars_array = MD2.rvs(size=50, random_state=rndint+timeseed)

    #     #mutant3 modifiers
    #     path3 = f'../data/smc_SM_hill/{mut3}_smc/pars_final.out'  #change final if needed
    #     df3 = Out_to_DF_hill(path2, model_hill.model_muts, mut3, all=False)
    #     MD3 = multivariate_dis(df3)
    #     mut3_pars_array = MD3.rvs(size=50, random_state=rndint+timeseed)

    #     #plot pairwise fit
    #     #selects mutant shortcode and assembles into correct mutant ID
    #     for i in range(0,10):
    #         if mut1.endswith(f'{i}'):
    #             if i == 0:
    #                 trip1 = f'{mut1[0]}1{i}'
    #             else:
    #                 trip1 = f'{mut1[0]}{i}'

    #         if mut2.endswith(f'{i}'):
    #             if i == 0:
    #                 trip2 = f'{mut2[0]}1{i}'
    #             else:
    #                 trip2 = f'{mut2[0]}{i}'
    #         if mut3.endswith(f'{i}'):
    #             if i == 0:
    #                 trip3 = f'{mut3[0]}1{i}'
    #             else:
    #                 trip3 = f'{mut3[0]}{i}'

    #     TM_df = meta_dict['TM']

    #     if trip1.startswith('R') | trip1.startswith('S'):

    #         trip_mut_dict = TM_df.loc[TM_df['genotype'].str.contains(trip1)]
    #         if trip1.endswith('1'):
    #             trip_mut_dict = trip_mut_dict.loc[trip_mut_dict['genotype'].str.contains('1_')]

    #     elif trip1.startswith('O'):
    #         trip_mut_dict = TM_df.loc[TM_df['genotype'].str.contains(trip1)]

    #         if trip1.endswith('1'):
    #             trip_mut_dict = trip_mut_dict.loc[trip_mut_dict['genotype'].str.endswith('1')]
    #             #incase pair ends with a 1 and 10 is included
    #     ####
    #     if trip2.startswith('R') | trip2.startswith('S'):

    #         trip_mut_dict = trip_mut_dict.loc[trip_mut_dict['genotype'].str.contains(trip2)]
    #         if trip2.endswith('1'):
    #             trip_mut_dict = trip_mut_dict.loc[trip_mut_dict['genotype'].str.contains('1_')]
                
    #     elif trip2.startswith('O'):
    #         trip_mut_dict = trip_mut_dict.loc[trip_mut_dict['genotype'].str.contains(trip2)]

    #         if trip2.endswith('1'):
    #             trip_mut_dict = trip_mut_dict.loc[trip_mut_dict['genotype'].str.endswith('1')]
    #             #incase pair ends with a 1 and 10 is included

    #     if trip3.startswith('R') | trip3.startswith('S'):

    #         trip_mut_dict = trip_mut_dict.loc[trip_mut_dict['genotype'].str.contains(trip3)]
    #         if trip3.endswith('1'):
    #             trip_mut_dict = trip_mut_dict.loc[trip_mut_dict['genotype'].str.contains('1_')]
                
    #     elif trip3.startswith('O'):
    #         trip_mut_dict = trip_mut_dict.loc[trip_mut_dict['genotype'].str.contains(trip3)]

    #         if trip3.endswith('1'):
    #             trip_mut_dict = trip_mut_dict.loc[trip_mut_dict['genotype'].str.endswith('1')]
    #             #incase pair ends with a 1 and 10 is included
        

    #     tripwise_fluo = []
    #     for fluo in trip_mut_dict['obs_fluo_mean']:
    #         tripwise_fluo.append(fluo)

    #     tripwise_inducer = [0.00001, 0.0002, 0.2]

    #     ind = pd.DataFrame(tripwise_inducer)


    #     plt.plot(tripwise_inducer,tripwise_fluo)
    #     plt.scatter(tripwise_inducer,tripwise_fluo)
    #     plt.xscale('log')
    #     plt.yscale('log')
    #     #plot prediction
        
    #     hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)
        
    #     for WT_pars,Mut1_pars,Mut2_pars, Mut3_pars in zip(WT_pars_array,mut1_pars_array,mut2_pars_array, mut3_pars_array):

    #         #identification of mutant types
    #         if mut1.startswith('Sensor') & mut2.startswith('Output'):
    #             M = {'As':Mut1_pars[0],'Bs':Mut1_pars[1],'Cs':Mut1_pars[2],'Ns':Mut1_pars[3],'Ar':Mut3_pars[0],'Br':Mut3_pars[1],'Cr':Mut3_pars[2],'Nr':Mut3_pars[3],'Ao':Mut2_pars[0],'Bo':Mut2_pars[1],'Co':Mut2_pars[2],'No':Mut2_pars[3],'Fo':0.0}
    #         elif mut1.startswith('Sensor') & mut2.startswith('Regulator'):
    #             M = {'As':Mut1_pars[0],'Bs':Mut1_pars[1],'Cs':Mut1_pars[2],'Ns':Mut1_pars[3],'Ar':Mut2_pars[0],'Br':Mut2_pars[1],'Cr':Mut2_pars[2],'Nr':Mut2_pars[3],'Ao':Mut3_pars[0],'Bo':Mut3_pars[1],'Co':Mut3_pars[2],'No':Mut3_pars[3],'Fo':0.0}
    #         elif mut1.startswith('Regulator') & mut2.startswith('Output'):
    #             M = {'As':Mut3_pars[0],'Bs':Mut3_pars[1],'Cs':Mut3_pars[2],'Ns':Mut3_pars[3],'Ar':Mut3_pars[0],'Br':Mut1_pars[1],'Cr':Mut1_pars[2],'Nr':Mut1_pars[3],'Ao':Mut2_pars[0],'Bo':Mut2_pars[1],'Co':Mut2_pars[2],'No':Mut2_pars[3],'Fo':0.0}
    #         elif mut1.startswith('Regulator') & mut2.startswith('Sensor'):
    #             M = {'As':Mut2_pars[0],'Bs':Mut2_pars[1],'Cs':Mut2_pars[2],'Ns':Mut2_pars[3],'Ar':Mut1_pars[1],'Br':Mut1_pars[1],'Cr':Mut1_pars[1],'Nr':Mut1_pars[1],'Ao':Mut3_pars[0],'Bo':Mut3_pars[1],'Co':Mut3_pars[2],'No':Mut3_pars[3],'Fo':0.0}
    #         elif mut1.startswith('Output') & mut2.startswith('Regulator'):
    #             M = {'As':Mut3_pars[0],'Bs':Mut3_pars[1],'Cs':Mut3_pars[2],'Ns':Mut3_pars[3],'Ar':Mut2_pars[0],'Br':Mut2_pars[1],'Cr':Mut2_pars[2],'Nr':Mut2_pars[3],'Ao':Mut1_pars[0],'Bo':Mut1_pars[1],'Co':Mut1_pars[2],'No':Mut1_pars[3],'Fo':0.0}
    #         elif mut1.startswith('Output') & mut2.startswith('Sensor'):
    #             M = {'As':Mut2_pars[0],'Bs':Mut2_pars[1],'Cs':Mut2_pars[2],'Ns':Mut2_pars[3],'Ar':Mut3_pars[0],'Br':Mut3_pars[1],'Cr':Mut3_pars[2],'Nr':Mut3_pars[3],'Ao':Mut1_pars[0],'Bo':Mut1_pars[1],'Co':Mut1_pars[2],'No':Mut1_pars[3],'Fo':0.0}
    #         else:
    #             raise KeyError('Mutant names invalid 212')
    #         par_dict = {
    #             "A_s":10**WT_pars[0],
    #             "B_s":10**WT_pars[1],
    #             "C_s":10**WT_pars[2],
    #             "N_s":WT_pars[3],
    #             "MA_s":10**M['As'],
    #             "MB_s":10**M['Bs'],
    #             "MC_s":10**M['Cs'],
    #             "MN_s":10**M['Ns'], 
    #             "A_r":10**WT_pars[4],
    #             "B_r":10**WT_pars[5],
    #             "C_r":10**WT_pars[6],
    #             "N_r":WT_pars[7],
    #             "MA_r":10**M['Ar'],
    #             "MB_r":10**M['Br'],
    #             "MC_r":10**M['Cr'],
    #             "MN_r":10**M['Nr'],
    #             "A_o":10**WT_pars[8],
    #             "B_o":10**WT_pars[9],
    #             "C_o":10**WT_pars[10],
    #             "N_o":WT_pars[11],
    #             "F_o":WT_pars[12],
    #             "MA_o":10**M['Ao'],
    #             "MB_o":10**M['Bo'],
    #             "MC_o":10**M['Co'],
    #             "MN_o":10**M['No'],
    #             "MF_o":10**M['Fo'],
    #                 }
            
    #         par_list = list(par_dict.values())


    #         Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model_muts(I_conc= ind,params_list=par_list)

    #         plt.plot(ind, Stripe_est_array, c = 'green', alpha = 0.1)
    #         plt.scatter(ind, Stripe_est_array, c = 'green', alpha = 0.1)

    #         plt.title(f'Pairwise mutant fit: {trip1}_{trip2}_{trip3}')

    # else:
    #     raise KeyError('incorrect number of mutants, must be 2 or 3')