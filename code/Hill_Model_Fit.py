#%%
'''Modules and other python files'''
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
import numpy as np
from scipy.stats import gaussian_kde
from random import seed
import time
import re
import seaborn as sns
from random import seed
import time
from Plotting_functions import *
#%%
'''Visualising Hill WT parameter distributions'''

# path = '../data/smc_hill/pars_final.out'

# WT_converged_params = Out_to_DF_hill(path, model_hill.model, '', all = False)

# param_dist = multivariate_dis(WT_converged_params)

# WT = 'Wildtype'
# Paired_Density_plot(WT_converged_params, name = WT, save=True)

#%%
'''Visualising whether the random selection of params from multi-v guassian is correct'''
# start_time_per_mutant=time.time()
# np.random.seed(0)  
# rndint = np.random.randint(low=0, high=1e7)
    
# timeseed = time.time_ns() % 2**16
# np.random.seed(rndint+timeseed)
# seed(rndint+timeseed)

# WT_converged_params = Out_to_DF_hill(path, model_hill.model, '', all = False) #WT SMC_ABC results

# param_dist = multivariate_dis(WT_converged_params) #convert to multivariable distribution

# random_params = param_dist.rvs(size=1, random_state=rndint+timeseed) #randomly selects one set of parameters

# #test to see if sampling from multi-variate distribution works
# param1 = []
# param1 = WT_converged_params['Br'].to_numpy()

# param2 = []
# param2 = WT_converged_params['Cr'].to_numpy()

# param1_gaus = []
# param2_gaus = []
# temp = param_dist.rvs(size=100, random_state=rndint+timeseed+200)

# for items in temp:
#     param1_gaus.append(items[5]) #Looks at the 5th and 6th parameters from the param set
#     param2_gaus.append(items[6]) #Both values were taken from the same random sample

# plt.scatter(param1,param2, c = 'b', label= 'WT param dist') #WT
# plt.scatter(param1_gaus,param2_gaus, c ='r', label= 'Multi-variate gaus') 
# %%
'''Plotting functions to visualise Hill modifier params for SM'''
#Visualising parameter distribution

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
    Stripe.scatter(data_.S, data_.Stripe, c = 'teal')
    Stripe.set_xscale('log')
    Stripe.set_yscale('log')
    Stripe.set_title(r'Full circuit with stripe')
    Output.scatter(data_.S, data_.Output, c = 'steelblue')
    Output.set_title(r'inducer -> S -| Output (GFP)')
    Output.set_xscale('log')
    Output.set_yscale('log')
    Regulator.scatter(data_.S, data_.Regulator, c = 'dimgrey')
    Regulator.set_xscale('log')
    Regulator.set_yscale('log')
    Regulator.set_title(r'inducer ->S -|R (GFP output)')
    Sensor.scatter(data_.S, data_.Sensor, c = 'darkorange')
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

    #Keeps track of all RSS scores in a list that can be compared to the par_array    
    score_list = []
    for i in range(0,len(par_array)):
        Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model_muts(I_conc=data_.S,params_list=par_array[i])

        Stripe.plot(data_.S, Stripe_est_array, alpha = 0.1, c = 'teal')
        Output.plot(data_.S, Output_est_array, alpha = 0.1, c = 'steelblue')
        Regulator.plot(data_.S, Regulator_est_array,alpha = 0.1, c = 'dimgrey')
        Sensor.plot(data_.S, Sensor_est_array, alpha = 0.1, c = 'darkorange')
        
        s = RSS_Score(par_array[i],model_hill,data_,model_specs='model_muts')
        score_list.append(round(s,3))

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

def Visualise_SM_par_dis(mut_name, iter, s:bool):
    path2 = f'../data/smc_SM_hill/{mut_name}_smc/pars_{iter}.out'  #only modifiers
    df2 = Out_to_DF_hill(path2, model_hill.model_muts, mut_name, all=False)
    
    return Paired_Density_plot_mut(df2, name = mut_name, save = s)

#put a mutant name and the final or last SMC step
mut = 'Sensor2'
num = 'final'
fig, scores, set_list = Visualise_SM_fit(mut_name=mut, iter = num, plot_num = 50, save = True)
# Visualise_SM_par_dis(mut_name= mut, iter = num, saves = True)

#%%
'''Visualise WT fittings'''
def Visualise_WT_fits(plot_num):
    hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)
    path = '../data/smc_hill/pars_final.out'
    df = Out_to_DF_hill(path, model_hill, mut_name= "", all = False)
    data_ = meta_dict['WT']
    plot_num = 50
    fig, ((Sensor, Regulator), (Output, Stripe)) = plt.subplots(2,2, constrained_layout=True)
    Stripe.scatter(data_.S, data_.Stripe, c = 'teal')
    Stripe.set_xscale('log')
    Stripe.set_yscale('log')
    Stripe.set_title(r'Full circuit with stripe')
    Output.scatter(data_.S, data_.Output, c = 'steelblue')
    Output.set_title(r'inducer -> S -| Output (GFP)')
    Output.set_xscale('log')
    Output.set_yscale('log')
    Regulator.scatter(data_.S, data_.Regulator, c = 'dimgrey')
    Regulator.set_xscale('log')
    Regulator.set_yscale('log')
    Regulator.set_title(r'inducer ->S -|R (GFP output)')
    Sensor.scatter(data_.S, data_.Sensor, c = 'darkorange')
    Sensor.set_xscale('log')
    Sensor.set_yscale('log')
    Sensor.set_title(r'inducer -> sensor (GFP output)')
    set_list = []
    par_array = np.empty([plot_num,13])
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

    for i in range(0,len(par_array)):
        Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.modelWT(I_conc=data_.S,params_list=par_array[i])

        Stripe.plot(data_.S, Stripe_est_array, alpha = 0.1, c = 'teal')
        Output.plot(data_.S, Output_est_array, alpha = 0.1, c = 'steelblue')
        Regulator.plot(data_.S, Regulator_est_array,alpha = 0.1, c = 'dimgrey')
        Sensor.plot(data_.S, Sensor_est_array, alpha = 0.1, c = 'darkorange')
    return
#%%

'''Examining the WT distribution of parameter sets between single mutants'''
def get_combo_WT_df_DM(mutants:list):
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

def get_combo_WT_df_TM(mutants:list):
    mutant1 = mutants[0]
    mutant2 = mutants[1]
    mutant3 = mutants[2]

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

    path3 = f'../data/smc_SM_hill/{mutant3}_smc/all_pars_final.out'  
    df3 = Out_to_DF_hill(path3, model_hill.model_muts, mutant3, all=True)
    WT3_df = df3[['As','Bs','Cs','Ns','Ar','Br','Cr','Nr','Ao','Bo','Co','No','Fo']]
    WT3_df = np.log10(WT3_df) #convert to log10
    mod_path3 = f'../data/smc_SM_hill/{mutant3}_smc/pars_final.out' 
    M3_mods_df = Out_to_DF_hill(mod_path3, model_hill.model_muts, mutant3, all=False)
    #a df with modifier params after wildtype

    M3_mods_df.reset_index(drop=True, inplace=True)
    WT3_df.reset_index(drop=True, inplace=True)
    M3_df = pd.concat([M3_mods_df, WT3_df,], axis=1)

    Combined_WT = pd.concat([WT1_df,WT2_df, WT3_df], axis=0)


    return Combined_WT, M1_mods_df, M2_mods_df, M3_mods_df, M1_df, M2_df, M3_df #all dfs are in log

# WT1_df = WT1_df.assign(Genotype= f'{mutant1}')
# WT2_df = WT2_df.assign(Genotype= f'{mutant2}')
#Paired_Density_plot_compare(Combined_WT,n,huw = 'Genotype', save=False)
# %%
'''Code to create a viable set of combined single mutant parameters'''
#Pairwise
def get_combo_params(mutants:list):
    '''Pairwise mutant combination of parameters'''
    #Step1: obtain mu of WT, mut1, mut2 and Covariance matrix of combined wildtypes

    Combined_WT, M1_mods_df, M2_mods_df, M1_df, M2_df = get_combo_WT_df_DM(mutants) #all log

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
    
    M1_cond_params = M1_multi_dis.rvs(size = 100, random_state=rndint+ timeseed)
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
    
    M2_cond_params = M2_multi_dis.rvs(size = 100, random_state=rndint+ timeseed)

    return WT_sample, M1_cond_params, M2_cond_params
#Centralised mean pairwise
def get_combo_params_0(mutants:list):
    '''Combine modifiers with centralised means'''
    #Step1: obtain mu of WT, mut1, mut2 and Covariance matrix of combined wildtypes

    Combined_WT, M1_mods_df, M2_mods_df, M1_df, M2_df = get_combo_WT_df_DM(mutants) #all log

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
    mu1 = [0,0,0,0]
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
    conditional_mu = mu1

    conditional_cov = C11 - np.dot(C12C22inv, C21)

    M1_multi_dis = multivariate_normal(mean = conditional_mu,
                                        cov = conditional_cov, 
                                        allow_singular = True
                                                 )
    
    M1_cond_params = M1_multi_dis.rvs(size = 100, random_state=rndint+ timeseed)
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
    mu1 = [0,0,0,0]
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
    conditional_mu = mu1

    conditional_cov = C11 - np.dot(C12C22inv, C21)

    M2_multi_dis = multivariate_normal(mean = conditional_mu,
                                       cov = conditional_cov,
                                       allow_singular = True
                                                 )
    
    M2_cond_params = M2_multi_dis.rvs(size = 100, random_state=rndint+ timeseed)

    return WT_sample, M1_cond_params, M2_cond_params
#Triplet
def get_combo_params_TM(mutants:list):
    '''Triplet combination of modifiers'''
    #Step1: obtain mu of WT, mut1, mut2 and Covariance matrix of combined wildtypes

    Combined_WT, M1_mods_df, M2_mods_df, M3_mods_df, M1_df, M2_df, M3_df = get_combo_WT_df_TM(mutants) #all log

    names = Combined_WT.keys()
    params = len(Combined_WT.columns)
    WT_matrix = np.empty(shape=(params,3000), dtype=float)
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
    
    M1_cond_params = M1_multi_dis.rvs(size = 100, random_state=rndint+ timeseed)
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
    
    M2_cond_params = M2_multi_dis.rvs(size = 100, random_state=rndint+ timeseed)
    ##################################################
    #Calculate mean for mut3 (just modifiers)
    names = M3_mods_df.keys()
    params = len(M3_mods_df.columns)
    M3_mods_matrix = np.empty(shape=(params,1000), dtype=float)
    i = 0
    for name in names:
        M3_mods_matrix[i] = M3_mods_df[name].to_numpy()
        i = i+1
    
    M3_mean_list = []
    j = 0

    for m in M3_mods_matrix:
        means = sum(m)
        means = means/len(m)
        M3_mean_list.append(means)
        j = j+1

    #Generate covariance matrix of full mutant params
    names = M3_df.keys()
    params = len(M3_df.columns)
    M3_matrix = np.empty(shape=(params,1000), dtype=float)
    i = 0
    for name in names:
        M3_matrix[i] = M3_df[name].to_numpy()
        i = i+1
    M3_df = M3_df.T
    M3_cov_matrix = np.cov(M3_df.values, bias = True)
    mu1 = M3_mean_list
    mu2 = WT_mean_list
    C11 = M3_cov_matrix[0:4,0:4]
    C12 = M3_cov_matrix[0:4:,4:]
    C21 = M3_cov_matrix[4:,0:4]
    C22 = M3_cov_matrix[4:,4:]
    C22inv = np.linalg.inv(C22)
    a_minus_mu = (WT_sample - mu2)
    a_minus_mu[:, np.newaxis]
    C12C22inv = np.dot(C12,C22inv.T) 
    temp = np.dot(C12C22inv, a_minus_mu[:, np.newaxis])
    conditional_mu = [x+y for x, y in zip(mu1,temp.flatten().tolist())]

    conditional_cov = C11 - np.dot(C12C22inv, C21)

    M3_multi_dis = multivariate_normal(mean = conditional_mu,
                                       cov = conditional_cov,
                                       allow_singular = True
                                                 )
    
    M3_cond_params = M3_multi_dis.rvs(size = 100, random_state=rndint+ timeseed)

    return WT_sample, M1_cond_params, M2_cond_params, M3_cond_params
#Centralised mean triplet
def get_combo_params_TM_0(mutants:list):

    #Step1: obtain mu of WT, mut1, mut2 and Covariance matrix of combined wildtypes

    Combined_WT, M1_mods_df, M2_mods_df, M3_mods_df, M1_df, M2_df, M3_df = get_combo_WT_df_TM(mutants) #all log

    names = Combined_WT.keys()
    params = len(Combined_WT.columns)
    WT_matrix = np.empty(shape=(params,3000), dtype=float)
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
    mu1 = [0,0,0,0]
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
    conditional_mu = mu1

    conditional_cov = C11 - np.dot(C12C22inv, C21)

    M1_multi_dis = multivariate_normal(mean = conditional_mu,
                                        cov = conditional_cov, 
                                        allow_singular = True
                                                 )
    
    M1_cond_params = M1_multi_dis.rvs(size = 100, random_state=rndint+ timeseed)
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
    mu1 = [0,0,0,0]
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
    conditional_mu = mu1

    conditional_cov = C11 - np.dot(C12C22inv, C21)

    M2_multi_dis = multivariate_normal(mean = conditional_mu,
                                       cov = conditional_cov,
                                       allow_singular = True
                                                 )
    
    M2_cond_params = M2_multi_dis.rvs(size = 100, random_state=rndint+ timeseed)
    ##################################################
    #Calculate mean for mut3 (just modifiers)
    names = M3_mods_df.keys()
    params = len(M3_mods_df.columns)
    M3_mods_matrix = np.empty(shape=(params,1000), dtype=float)
    i = 0
    for name in names:
        M3_mods_matrix[i] = M3_mods_df[name].to_numpy()
        i = i+1
    
    M3_mean_list = []
    j = 0

    for m in M3_mods_matrix:
        means = sum(m)
        means = means/len(m)
        M3_mean_list.append(means)
        j = j+1

    #Generate covariance matrix of full mutant params
    names = M3_df.keys()
    params = len(M3_df.columns)
    M3_matrix = np.empty(shape=(params,1000), dtype=float)
    i = 0
    for name in names:
        M3_matrix[i] = M3_df[name].to_numpy()
        i = i+1
    M3_df = M3_df.T
    M3_cov_matrix = np.cov(M3_df.values, bias = True)
    mu1 = [0,0,0,0]
    mu2 = WT_mean_list
    C11 = M3_cov_matrix[0:4,0:4]
    C12 = M3_cov_matrix[0:4:,4:]
    C21 = M3_cov_matrix[4:,0:4]
    C22 = M3_cov_matrix[4:,4:]
    C22inv = np.linalg.inv(C22)
    a_minus_mu = (WT_sample - mu2)
    a_minus_mu[:, np.newaxis]
    C12C22inv = np.dot(C12,C22inv.T) 
    temp = np.dot(C12C22inv, a_minus_mu[:, np.newaxis])
    conditional_mu = mu1

    conditional_cov = C11 - np.dot(C12C22inv, C21)

    M3_multi_dis = multivariate_normal(mean = conditional_mu,
                                       cov = conditional_cov,
                                       allow_singular = True
                                                 )
    
    M3_cond_params = M3_multi_dis.rvs(size = 100, random_state=rndint+ timeseed)

    return WT_sample, M1_cond_params, M2_cond_params, M3_cond_params
# mutants = ['Regulator9','Output3']
# WT, M1, M2 = get_combo_params(mutants)
# %%
'''Visualisation of pairwise/triplet fits using combined modifiers'''
def Visualise_combo_mut_fit(mutants:list):
    '''Takes 2 or 3 mutants in a list and plots them to the data, enter mutants as follows [Output/Sensor/Regulator[1-10]]'''
    if len(mutants) == 2:
        mut1 = mutants[0]
        mut2 = mutants[1]

        rndint = np.random.randint(low=0, high=1e7)
        timeseed = time.time_ns() % 2**16
        np.random.seed(rndint+timeseed)
        seed(rndint+timeseed)

        size = 1000
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
                pair_mut_dict = pair_mut_dict.loc[pair_mut_dict['genotype'].str.endswith('1')]
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


        #plot mutant fits
        hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)

        WT_pars_array = np.empty(shape=(size,13))
        Mut1_pars_array = np.empty(shape=(size,4))
        Mut2_pars_array = np.empty(shape=(size,4))

        low = []
        med = []
        high = []
        parameters = pd.DataFrame({
                    "A_s":[],
                    "B_s":[],
                    "C_s":[],
                    "N_s":[],
                    "MA_s":[],
                    "MB_s":[],
                    "MC_s":[],
                    "MN_s":[], 
                    "A_r":[],
                    "B_r":[],
                    "C_r":[],
                    "N_r":[],
                    "MA_r":[],
                    "MB_r":[],
                    "MC_r":[],
                    "MN_r":[],
                    "A_o":[],
                    "B_o":[],
                    "C_o":[],
                    "N_o":[],
                    "F_o":[],
                    "MA_o":[],
                    "MB_o":[],
                    "MC_o":[],
                    "MN_o":[],
                    "MF_o":[],
                        })
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
                    "N_s":WT_pars[3],
                    "MA_s":10**M['As'],
                    "MB_s":10**M['Bs'],
                    "MC_s":10**M['Cs'],
                    "MN_s":10**M['Ns'], 
                    "A_r":10**WT_pars[4],
                    "B_r":10**WT_pars[5],
                    "C_r":10**WT_pars[6],
                    "N_r":WT_pars[7],
                    "MA_r":10**M['Ar'],
                    "MB_r":10**M['Br'],
                    "MC_r":10**M['Cr'],
                    "MN_r":10**M['Nr'],
                    "A_o":10**WT_pars[8],
                    "B_o":10**WT_pars[9],
                    "C_o":10**WT_pars[10],
                    "N_o":WT_pars[11],
                    "F_o":WT_pars[12],
                    "MA_o":10**M['Ao'],
                    "MB_o":10**M['Bo'],
                    "MC_o":10**M['Co'],
                    "MN_o":10**M['No'],
                    "MF_o":10**M['Fo'],
                        }
            
                par_list = list(par_dict.values())

                parameters.loc[len(parameters)] = par_list

                Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model_muts(I_conc= ind,params_list=par_list)



                low.append(Stripe_est_array.iloc[0,0])
                med.append(Stripe_est_array.iloc[1,0])
                high.append(Stripe_est_array.iloc[2,0]) 

                
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

        Rand = mpatches.Patch(color= 'mistyrose', label='Estimated fluorescence')
        Wildtype = mpatches.Patch(color= 'indigo', label='Wildtype') #Could potenitally plot the actual wildtype data
        data_set = mpatches.Patch(color= 'darkcyan', label='Pairwise data')
        plt.legend(handles=[data_set,Rand,Wildtype], bbox_to_anchor=(1, 1), title = "Legend")
        plt.title(f'Pairwise mutant fit: {pair1}_{pair2}')
        axes.set_xlabel('Inducer Concetration')
        axes.set_ylim(6,10)
        axes2.set_ylim(6,10)
        axes.set_ylabel('Log_Fluorescence')
        axes.set_xticks(ticks=range(len(fluo_df.columns)), labels=fluo_df.columns)


    elif len(mutants) == 3:
        mut1 = mutants[0]
        mut2 = mutants[1]
        mut3 = mutants[2]

        size = 100
        rndint = np.random.randint(low=0, high=1e7)
        timeseed = time.time_ns() % 2**16
        np.random.seed(rndint+timeseed)
        seed(rndint+timeseed)

        #plot pairwise fit
        #selects mutant shortcode and assembles into correct mutant ID

        TM_df = meta_dict['TM']

        #All mutants are in order of R_S_O

        mutid = get_mut_ids(mutants)

        trip_mut_dict = TM_df[TM_df['genotype'] == mutid]
        

        tripwise_fluo = []
        for fluo in trip_mut_dict['obs_fluo_mean']:
            tripwise_fluo.append(fluo)


        tripwise_inducer = [0.00001, 0.0002, 0.2]

        ind = pd.DataFrame(tripwise_inducer)
        #plot mutant fits
        hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)

        WT_pars_array = np.empty(shape=(size,13))
        Mut1_pars_array = np.empty(shape=(size,4))
        Mut2_pars_array = np.empty(shape=(size,4))
        Mut3_pars_array = np.empty(shape=(size,4))

        low = []
        med = []
        high = []

        for i in range(0,size):
            WT_pars, Mut1_pars_array, Mut2_pars_array, Mut3_pars_array = get_combo_params_TM(mutants)
            for Mut1_pars,Mut2_pars,Mut3_pars in zip(Mut1_pars_array,Mut2_pars_array, Mut3_pars_array):
            #identification of mutant types
                if mut1.startswith('Sensor') & mut2.startswith('Output'):
                    M = {'As':Mut1_pars[0],'Bs':Mut1_pars[1],'Cs':Mut1_pars[2],'Ns':Mut1_pars[3],'Ar':Mut3_pars[0],'Br':Mut3_pars[1],'Cr':Mut3_pars[2],'Nr':Mut3_pars[3],'Ao':Mut2_pars[0],'Bo':Mut2_pars[1],'Co':Mut2_pars[2],'No':Mut2_pars[3],'Fo':0.0}
                elif mut1.startswith('Sensor') & mut2.startswith('Regulator'):
                    M = {'As':Mut1_pars[0],'Bs':Mut1_pars[1],'Cs':Mut1_pars[2],'Ns':Mut1_pars[3],'Ar':Mut2_pars[0],'Br':Mut2_pars[1],'Cr':Mut2_pars[2],'Nr':Mut2_pars[3],'Ao':Mut3_pars[0],'Bo':Mut3_pars[1],'Co':Mut3_pars[2],'No':Mut3_pars[3],'Fo':0.0}
                elif mut1.startswith('Regulator') & mut2.startswith('Output'):
                    M = {'As':Mut3_pars[0],'Bs':Mut3_pars[1],'Cs':Mut3_pars[2],'Ns':Mut3_pars[3],'Ar':Mut1_pars[0],'Br':Mut1_pars[1],'Cr':Mut1_pars[2],'Nr':Mut1_pars[3],'Ao':Mut2_pars[0],'Bo':Mut2_pars[1],'Co':Mut2_pars[2],'No':Mut2_pars[3],'Fo':0.0}
                elif mut1.startswith('Regulator') & mut2.startswith('Sensor'):
                    M = {'As':Mut2_pars[0],'Bs':Mut2_pars[1],'Cs':Mut2_pars[2],'Ns':Mut2_pars[3],'Ar':Mut1_pars[1],'Br':Mut1_pars[1],'Cr':Mut1_pars[1],'Nr':Mut1_pars[1],'Ao':Mut3_pars[0],'Bo':Mut3_pars[1],'Co':Mut3_pars[2],'No':Mut3_pars[3],'Fo':0.0}
                elif mut1.startswith('Output') & mut2.startswith('Regulator'):
                    M = {'As':Mut3_pars[0],'Bs':Mut3_pars[1],'Cs':Mut3_pars[2],'Ns':Mut3_pars[3],'Ar':Mut2_pars[0],'Br':Mut2_pars[1],'Cr':Mut2_pars[2],'Nr':Mut2_pars[3],'Ao':Mut1_pars[0],'Bo':Mut1_pars[1],'Co':Mut1_pars[2],'No':Mut1_pars[3],'Fo':0.0}
                elif mut1.startswith('Output') & mut2.startswith('Sensor'):
                    M = {'As':Mut2_pars[0],'Bs':Mut2_pars[1],'Cs':Mut2_pars[2],'Ns':Mut2_pars[3],'Ar':Mut3_pars[0],'Br':Mut3_pars[1],'Cr':Mut3_pars[2],'Nr':Mut3_pars[3],'Ao':Mut1_pars[0],'Bo':Mut1_pars[1],'Co':Mut1_pars[2],'No':Mut1_pars[3],'Fo':0.0}
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

        data = {'low':np.log(low), 'medium':np.log(med), 'high':np.log(high)}
        fluo_df = pd.DataFrame(data)
        fig, axes = plt.subplots(figsize=(10,6))

        axes2 = axes.twinx()
        point = []
        SD = []
        for obs_m, obs_sd in zip(trip_mut_dict['obs_fluo_mean'],trip_mut_dict['obs_SD']):
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

        Rand = mpatches.Patch(color= 'mistyrose', label='Estimated fluorescence')
        Wildtype = mpatches.Patch(color= 'indigo', label='Wildtype') #Could potenitally plot the actual wildtype data
        data_set = mpatches.Patch(color= 'darkcyan', label='Pairwise data')
        plt.legend(handles=[data_set,Rand,Wildtype], bbox_to_anchor=(1, 1), title = "Legend")
        plt.title(f'Triplet mutant fit: {mutid}')
        axes.set_xlabel('Inducer Concetration')
        axes.set_ylim(6,9)
        axes2.set_ylim(6,9)
        axes.set_ylabel('Log_Fluorescence')
        axes.set_xticks(ticks=range(len(fluo_df.columns)), labels=fluo_df.columns)
# mutants = ['Regulator10', 'Sensor10', 'Output10']
# Visualise_combo_mut_fit(mutants)
# %%
'''calculating the predicted fluorescence from parameter sets'''
def obtain_pred_fluo(mutants:list, size = 100):
    '''Takes 2 or 3 mutants in a list and returns a dataframe of 3*size*1000 for predicted fluorescence at low, medium and high inducer concs '''
    if len(mutants) == 2:
        mut1 = mutants[0]
        mut2 = mutants[1]

        rndint = np.random.randint(low=0, high=1e7)
        timeseed = time.time_ns() % 2**16
        np.random.seed(rndint+timeseed)
        seed(rndint+timeseed)

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
                pair_mut_dict = pair_mut_dict.loc[pair_mut_dict['genotype'].str.endswith('1')]
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

        genotype = [f'{pair1}_{pair2}']*len(low)

        temp = {'Genotype':genotype, 'low':low, 'medium':med, 'high':high}

        df = pd.DataFrame(temp)

        return df
    elif len(mutants) == 3:
        mut1 = mutants[0]
        mut2 = mutants[1]
        mut3 = mutants[2]


        rndint = np.random.randint(low=0, high=1e7)
        timeseed = time.time_ns() % 2**16
        np.random.seed(rndint+timeseed)
        seed(rndint+timeseed)

        #plot pairwise fit
        #selects mutant shortcode and assembles into correct mutant ID

        TM_df = meta_dict['TM']

        #All mutants are in order of R_S_O

        mutid = get_mut_ids(mutants)

        trip_mut_dict = TM_df[TM_df['genotype'] == mutid]
        

        tripwise_fluo = []
        for fluo in trip_mut_dict['obs_fluo_mean']:
            tripwise_fluo.append(fluo)


        tripwise_inducer = [0.00001, 0.0002, 0.2]

        ind = pd.DataFrame(tripwise_inducer)
        #plot mutant fits
        hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)

        WT_pars_array = np.empty(shape=(size,13))
        Mut1_pars_array = np.empty(shape=(size,4))
        Mut2_pars_array = np.empty(shape=(size,4))
        Mut3_pars_array = np.empty(shape=(size,4))

        low = []
        med = []
        high = []

        for i in range(0,size):
            WT_pars, Mut1_pars_array, Mut2_pars_array, Mut3_pars_array = get_combo_params_TM(mutants)
            for Mut1_pars,Mut2_pars,Mut3_pars in zip(Mut1_pars_array,Mut2_pars_array, Mut3_pars_array):
            #identification of mutant types
                if mut1.startswith('Sensor') & mut2.startswith('Output'):
                    M = {'As':Mut1_pars[0],'Bs':Mut1_pars[1],'Cs':Mut1_pars[2],'Ns':Mut1_pars[3],'Ar':Mut3_pars[0],'Br':Mut3_pars[1],'Cr':Mut3_pars[2],'Nr':Mut3_pars[3],'Ao':Mut2_pars[0],'Bo':Mut2_pars[1],'Co':Mut2_pars[2],'No':Mut2_pars[3],'Fo':0.0}
                elif mut1.startswith('Sensor') & mut2.startswith('Regulator'):
                    M = {'As':Mut1_pars[0],'Bs':Mut1_pars[1],'Cs':Mut1_pars[2],'Ns':Mut1_pars[3],'Ar':Mut2_pars[0],'Br':Mut2_pars[1],'Cr':Mut2_pars[2],'Nr':Mut2_pars[3],'Ao':Mut3_pars[0],'Bo':Mut3_pars[1],'Co':Mut3_pars[2],'No':Mut3_pars[3],'Fo':0.0}
                elif mut1.startswith('Regulator') & mut2.startswith('Output'):
                    M = {'As':Mut3_pars[0],'Bs':Mut3_pars[1],'Cs':Mut3_pars[2],'Ns':Mut3_pars[3],'Ar':Mut1_pars[0],'Br':Mut1_pars[1],'Cr':Mut1_pars[2],'Nr':Mut1_pars[3],'Ao':Mut2_pars[0],'Bo':Mut2_pars[1],'Co':Mut2_pars[2],'No':Mut2_pars[3],'Fo':0.0}
                elif mut1.startswith('Regulator') & mut2.startswith('Sensor'):
                    M = {'As':Mut2_pars[0],'Bs':Mut2_pars[1],'Cs':Mut2_pars[2],'Ns':Mut2_pars[3],'Ar':Mut1_pars[1],'Br':Mut1_pars[1],'Cr':Mut1_pars[1],'Nr':Mut1_pars[1],'Ao':Mut3_pars[0],'Bo':Mut3_pars[1],'Co':Mut3_pars[2],'No':Mut3_pars[3],'Fo':0.0}
                elif mut1.startswith('Output') & mut2.startswith('Regulator'):
                    M = {'As':Mut3_pars[0],'Bs':Mut3_pars[1],'Cs':Mut3_pars[2],'Ns':Mut3_pars[3],'Ar':Mut2_pars[0],'Br':Mut2_pars[1],'Cr':Mut2_pars[2],'Nr':Mut2_pars[3],'Ao':Mut1_pars[0],'Bo':Mut1_pars[1],'Co':Mut1_pars[2],'No':Mut1_pars[3],'Fo':0.0}
                elif mut1.startswith('Output') & mut2.startswith('Sensor'):
                    M = {'As':Mut2_pars[0],'Bs':Mut2_pars[1],'Cs':Mut2_pars[2],'Ns':Mut2_pars[3],'Ar':Mut3_pars[0],'Br':Mut3_pars[1],'Cr':Mut3_pars[2],'Nr':Mut3_pars[3],'Ao':Mut1_pars[0],'Bo':Mut1_pars[1],'Co':Mut1_pars[2],'No':Mut1_pars[3],'Fo':0.0}
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

        genotype = [mutid]*len(low)

        temp = {'Genotype':genotype, 'low':low, 'medium':med, 'high':high}

        df = pd.DataFrame(temp)

        return df

def obtain_pred_fluo_0(mutants:list, size = 100):
    '''Takes 2 or 3 mutants in a list and returns a dataframe of 3*size*1000 for predicted fluorescence at low, medium and high inducer concs '''
    if len(mutants) == 2:
        mut1 = mutants[0]
        mut2 = mutants[1]

        rndint = np.random.randint(low=0, high=1e7)
        timeseed = time.time_ns() % 2**16
        np.random.seed(rndint+timeseed)
        seed(rndint+timeseed)

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
                pair_mut_dict = pair_mut_dict.loc[pair_mut_dict['genotype'].str.endswith('1')]
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
        #plot mutant fits
        hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)

        WT_pars_array = np.empty(shape=(size,13))
        Mut1_pars_array = np.empty(shape=(size,4))
        Mut2_pars_array = np.empty(shape=(size,4))

        low = []
        med = []
        high = []

        for i in range(0,size):
            WT_pars, Mut1_pars_array, Mut2_pars_array = get_combo_params_0(mutants)
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

        genotype = [f'{pair1}_{pair2}']*len(low)

        temp = {'Genotype':genotype, 'low':low, 'medium':med, 'high':high}

        df = pd.DataFrame(temp)

        return df
    elif len(mutants) == 3:
        mut1 = mutants[0]
        mut2 = mutants[1]
        mut3 = mutants[2]


        rndint = np.random.randint(low=0, high=1e7)
        timeseed = time.time_ns() % 2**16
        np.random.seed(rndint+timeseed)
        seed(rndint+timeseed)

        #plot pairwise fit
        #selects mutant shortcode and assembles into correct mutant ID

        TM_df = meta_dict['TM']

        #All mutants are in order of R_S_O

        mutid = get_mut_ids(mutants)

        trip_mut_dict = TM_df[TM_df['genotype'] == mutid]
        

        tripwise_fluo = []
        for fluo in trip_mut_dict['obs_fluo_mean']:
            tripwise_fluo.append(fluo)


        tripwise_inducer = [0.00001, 0.0002, 0.2]

        ind = pd.DataFrame(tripwise_inducer)
        #plot mutant fits
        hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)

        WT_pars_array = np.empty(shape=(size,13))
        Mut1_pars_array = np.empty(shape=(size,4))
        Mut2_pars_array = np.empty(shape=(size,4))
        Mut3_pars_array = np.empty(shape=(size,4))

        low = []
        med = []
        high = []

        for i in range(0,size):
            WT_pars, Mut1_pars_array, Mut2_pars_array, Mut3_pars_array = get_combo_params_TM_0(mutants)
            for Mut1_pars,Mut2_pars,Mut3_pars in zip(Mut1_pars_array,Mut2_pars_array, Mut3_pars_array):
            #identification of mutant types
                if mut1.startswith('Sensor') & mut2.startswith('Output'):
                    M = {'As':Mut1_pars[0],'Bs':Mut1_pars[1],'Cs':Mut1_pars[2],'Ns':Mut1_pars[3],'Ar':Mut3_pars[0],'Br':Mut3_pars[1],'Cr':Mut3_pars[2],'Nr':Mut3_pars[3],'Ao':Mut2_pars[0],'Bo':Mut2_pars[1],'Co':Mut2_pars[2],'No':Mut2_pars[3],'Fo':0.0}
                elif mut1.startswith('Sensor') & mut2.startswith('Regulator'):
                    M = {'As':Mut1_pars[0],'Bs':Mut1_pars[1],'Cs':Mut1_pars[2],'Ns':Mut1_pars[3],'Ar':Mut2_pars[0],'Br':Mut2_pars[1],'Cr':Mut2_pars[2],'Nr':Mut2_pars[3],'Ao':Mut3_pars[0],'Bo':Mut3_pars[1],'Co':Mut3_pars[2],'No':Mut3_pars[3],'Fo':0.0}
                elif mut1.startswith('Regulator') & mut2.startswith('Output'):
                    M = {'As':Mut3_pars[0],'Bs':Mut3_pars[1],'Cs':Mut3_pars[2],'Ns':Mut3_pars[3],'Ar':Mut1_pars[0],'Br':Mut1_pars[1],'Cr':Mut1_pars[2],'Nr':Mut1_pars[3],'Ao':Mut2_pars[0],'Bo':Mut2_pars[1],'Co':Mut2_pars[2],'No':Mut2_pars[3],'Fo':0.0}
                elif mut1.startswith('Regulator') & mut2.startswith('Sensor'):
                    M = {'As':Mut2_pars[0],'Bs':Mut2_pars[1],'Cs':Mut2_pars[2],'Ns':Mut2_pars[3],'Ar':Mut1_pars[1],'Br':Mut1_pars[1],'Cr':Mut1_pars[1],'Nr':Mut1_pars[1],'Ao':Mut3_pars[0],'Bo':Mut3_pars[1],'Co':Mut3_pars[2],'No':Mut3_pars[3],'Fo':0.0}
                elif mut1.startswith('Output') & mut2.startswith('Regulator'):
                    M = {'As':Mut3_pars[0],'Bs':Mut3_pars[1],'Cs':Mut3_pars[2],'Ns':Mut3_pars[3],'Ar':Mut2_pars[0],'Br':Mut2_pars[1],'Cr':Mut2_pars[2],'Nr':Mut2_pars[3],'Ao':Mut1_pars[0],'Bo':Mut1_pars[1],'Co':Mut1_pars[2],'No':Mut1_pars[3],'Fo':0.0}
                elif mut1.startswith('Output') & mut2.startswith('Sensor'):
                    M = {'As':Mut2_pars[0],'Bs':Mut2_pars[1],'Cs':Mut2_pars[2],'Ns':Mut2_pars[3],'Ar':Mut3_pars[0],'Br':Mut3_pars[1],'Cr':Mut3_pars[2],'Nr':Mut3_pars[3],'Ao':Mut1_pars[0],'Bo':Mut1_pars[1],'Co':Mut1_pars[2],'No':Mut1_pars[3],'Fo':0.0}
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

        genotype = [mutid]*len(low)

        temp = {'Genotype':genotype, 'low':low, 'medium':med, 'high':high}

        df = pd.DataFrame(temp)

        return df
'''Obtain the mode of a distribution of parameters'''
def Kde_mode(df):
    '''Finds the mode from a kde of low medium high data'''
    colnames = df.columns.values 
    low = df[colnames[1]].to_list() #assumes that first column is genotype
    med = df[colnames[2]].to_list()
    high = df[colnames[3]].to_list()

    low_kde = gaussian_kde(low)
    medium_kde = gaussian_kde(med)
    high_kde = gaussian_kde(high)

    x = np.linspace(min(low),max(low),num=1000)
    y = low_kde(x)

    mode_index = np.argmax(y)
    low_mode = x[mode_index]

    x = np.linspace(min(med),max(med),num=1000)
    y = medium_kde(x)

    mode_index = np.argmax(y)
    med_mode = x[mode_index]

    x = np.linspace(min(high),max(high),num=1000)
    y = high_kde(x)

    mode_index = np.argmax(y)
    high_mode = x[mode_index]

    return low_mode, med_mode, high_mode

#%%
'''Obtain all epistasis hat values for our model'''
from data_wrangling import *
df_DM = meta_dict['DM']
g_WT = np.array(df_DM['obs_fluo_mean'][df_DM['genotype']=='WT'])
def G_hat_all(df):
    low = df['low'].tolist()
    med =df['medium'].tolist()
    high = df['high'].tolist()
    
    l_g_hat = [np.divide(l,g_WT[0]) for l in low]
    m_g_hat = [np.divide(m,g_WT[1]) for m in med]
    h_g_hat = [np.divide(h,g_WT[2]) for h in high]

    G_low = [np.log10(ghat) for ghat in l_g_hat]
    G_med = [np.log10(ghat) for ghat in m_g_hat]
    G_high = [np.log10(ghat) for ghat in h_g_hat]
    
    genotype = df['Genotype'].tolist()

    temp = {'Genotype':genotype, 'G_low':G_low, 'G_medium':G_med, 'G_high':G_high}

    df2 = pd.DataFrame(temp)

    return df2
def Eps(df, Glog):
    Eps_low = []
    Eps_med = []
    Eps_high = []
    for l,m,h in zip(df['G_low'],df['G_medium'],df['G_high']):
        Ep_low = l - Glog[0]
        Eps_low.append(Ep_low)

        Ep_med = m - Glog[1]
        Eps_med.append(Ep_med)

        Ep_high = h - Glog[2]
        Eps_high.append(Ep_high)
    
    genotype = df['Genotype'].tolist()

    temp = {'Genotype':genotype, 'Eps_low':Eps_low, 'Eps_medium':Eps_med, 'Eps_high':Eps_high}

    df2 = pd.DataFrame(temp)

    return df2
def Generate_Eps_hat(mutant_list):
        '''Get epistasis values for a given fluroescence output per mutant '''
        #find G_log of mutants of interest
        Glog_mean= G_log(mutant_list)[0]

        #Get G_hat values for all predicted fluorescence 
        LMH_df = obtain_pred_fluo(mutant_list) #low,med,high fluoresence
        df = G_hat_all(LMH_df)
        
        return Eps(df,Glog_mean), Glog_mean

def Generate_Eps_hat_0(mutant_list):
        '''Get epistasis values for a given fluroescence output per mutant '''
        #find G_log of mutants of interest
        Glog_mean= G_log(mutant_list)[0]

        #Get G_hat values for all predicted fluorescence 
        LMH_df = obtain_pred_fluo_0(mutant_list) #low,med,high fluoresence
        df = G_hat_all(LMH_df)
        
        return Eps(df,Glog_mean), Glog_mean


#Calculate epistasis hat for low medium and high inducer conc
def Generate_Epshat_all_DM(prior_mutant:None):
    '''Create large dataframe of all Epistasis at low, medium and high inducer concs'''
    DM_names = DM_stripes['genotype'].tolist()
    DM_names = list(set(DM_names[3:]))
    
    if prior_mutant == None:
        Eps_pairwise_df = pd.DataFrame({'Genotype': [], 'Category':[], 'Epistasis_hat':[], 'Epistasis_obs': []})
        mutant_range:slice=slice(0,len(DM_names))
        Eps_pairwise_df.to_csv('../results/Pairwise_Eps.csv', index = False)
        path = f"../results/Pairwise_Eps.csv"
    else:
        position = DM_names.index(prior_mutant)
        path = f"../results/Pairwise_Eps.csv"
        Eps_pairwise_df = pd.read_csv(path)
        mutant_range:slice=slice(position,len(DM_names))

    for genotypes in DM_names[mutant_range]:
        #get genotypeID
        mutant_list = get_mut_names(genotypes)
        #Obtain epistasis values for predictions and LAD from LMH data 
        Eps_df, G_log_mean= Generate_Eps_hat(mutant_list)
        
        G = G_lab(mutant_list)
        G_obs = G[0] #observed fluoresence
        low_Epsilon =  G_obs[0] - G_log_mean[0] #G_log_mean from the LAD model
        med_Epsilon =  G_obs[1] - G_log_mean[1]
        high_Epsilon =  G_obs[2] - G_log_mean[2]

        Eps_hat = [] 
        Categories = []
        Eps_obs = []

        for items in Eps_df['Eps_low'].tolist():
            Eps_hat.append(items)
            Categories.append('low')
            Eps_obs.append(low_Epsilon)
        for items in Eps_df['Eps_medium'].tolist():
            Eps_hat.append(items)
            Categories.append('medium')
            Eps_obs.append(med_Epsilon)
        for items in Eps_df['Eps_high'].tolist():
            Eps_hat.append(items)
            Categories.append('high')
            Eps_obs.append(high_Epsilon)

        Genotype_list = Eps_df['Genotype'].tolist() * 3


        temp = {'Genotype': Genotype_list, 'Category':Categories, 'Epistasis_hat':Eps_hat, 'Epistasis_obs': Eps_obs}

        new_df = pd.DataFrame(temp)
        new_df.to_csv(path,mode='a', header=False, index=False)

        print(f'Mutant {genotypes} completed')
    
    print('mutants complete')
    return

def Generate_Epshat_all_TM(prior_mutant:None):
    '''Create large dataframe of all Epistasis at low, medium and high inducer concs'''
    TM_names = TM_stripes['genotype'].tolist()
    TM_names = list(set(TM_names[3:]))
    TM_names.sort()
    
    if prior_mutant == None:
        Eps_triplet_df = pd.DataFrame({'Genotype': [], 'Category':[], 'Epistasis_hat':[], 'Epistasis_obs': []})
        mutant_range:slice=slice(0,len(TM_names))
        Eps_triplet_df.to_csv('../results/Triplet_Eps.csv', index = False)
        path = f"../results/Triplet_Eps.csv"
    else:
        position = TM_names.index(prior_mutant)
        path = f"../results/Triplet_Eps.csv"
        Eps_triplet_df = pd.read_csv(path)
        mutant_range:slice=slice(position+1,len(TM_names))

    for genotypes in TM_names[mutant_range]:
        #get genotypeID
        mutant_list = get_mut_names(genotypes)
        #Obtain epistasis values for predictions and LAD from LMH data 
        Eps_df, G_log_mean= Generate_Eps_hat(mutant_list)
        
        G = G_lab(mutant_list)
        G_obs = G[0] #observed fluoresence
        low_Epsilon =  G_obs[0] - G_log_mean[0] #G_log_mean from the LAD model
        med_Epsilon =  G_obs[1] - G_log_mean[1]
        high_Epsilon =  G_obs[2] - G_log_mean[2]

        Eps_hat = [] 
        Categories = []
        Eps_obs = []

        for items in Eps_df['Eps_low'].tolist():
            Eps_hat.append(items)
            Categories.append('low')
            Eps_obs.append(low_Epsilon)
        for items in Eps_df['Eps_medium'].tolist():
            Eps_hat.append(items)
            Categories.append('medium')
            Eps_obs.append(med_Epsilon)
        for items in Eps_df['Eps_high'].tolist():
            Eps_hat.append(items)
            Categories.append('high')
            Eps_obs.append(high_Epsilon)

        Genotype_list = Eps_df['Genotype'].tolist() * 3


        temp = {'Genotype': Genotype_list, 'Category':Categories, 'Epistasis_hat':Eps_hat, 'Epistasis_obs': Eps_obs}

        new_df = pd.DataFrame(temp)
        new_df.to_csv(path,mode='a', header=False, index=False)

        # book = load_workbook(path)
        # writer = pd.ExcelWriter(path, engine='openpyxl')
        # writer.book = book
        # Eps_pairwise_df = pd.read_excel(path)
        # updated_data = pd.concat([Eps_pairwise_df,new_df],ignore_index=True)
        # updated_data.to_excel(writer, index = False)
        print(f'Mutant {genotypes} completed')
    
    print('mutants complete')
    return

def Generate_Epshat_all_DM_0(prior_mutant:None):
    '''Create large dataframe of all Epistasis at low, medium and high inducer concs'''
    DM_names = DM_stripes['genotype'].tolist()
    DM_names = list(set(DM_names[3:]))
    
    if prior_mutant == None:
        Eps_pairwise_df = pd.DataFrame({'Genotype': [], 'Category':[], 'Epistasis_hat':[], 'Epistasis_obs': []})
        mutant_range:slice=slice(0,len(DM_names))
        Eps_pairwise_df.to_csv('../results/Pairwise_Eps_0.csv', index = False)
        path = f"../results/Pairwise_Eps_0.csv"
    else:
        position = DM_names.index(prior_mutant)
        path = f"../results/Pairwise_Eps.csv"
        Eps_pairwise_df = pd.read_csv(path)
        mutant_range:slice=slice(position,len(DM_names))

    for genotypes in DM_names[mutant_range]:
        #get genotypeID
        mutant_list = get_mut_names(genotypes)
        #Obtain epistasis values for predictions and LAD from LMH data 
        Eps_df, G_log_mean= Generate_Eps_hat_0(mutant_list)
        
        G = G_lab(mutant_list)
        G_obs = G[0] #observed fluoresence
        low_Epsilon =  G_obs[0] - G_log_mean[0] #G_log_mean from the LAD model
        med_Epsilon =  G_obs[1] - G_log_mean[1]
        high_Epsilon =  G_obs[2] - G_log_mean[2]

        Eps_hat = [] 
        Categories = []
        Eps_obs = []

        for items in Eps_df['Eps_low'].tolist():
            Eps_hat.append(items)
            Categories.append('low')
            Eps_obs.append(low_Epsilon)
        for items in Eps_df['Eps_medium'].tolist():
            Eps_hat.append(items)
            Categories.append('medium')
            Eps_obs.append(med_Epsilon)
        for items in Eps_df['Eps_high'].tolist():
            Eps_hat.append(items)
            Categories.append('high')
            Eps_obs.append(high_Epsilon)

        Genotype_list = Eps_df['Genotype'].tolist() * 3


        temp = {'Genotype': Genotype_list, 'Category':Categories, 'Epistasis_hat':Eps_hat, 'Epistasis_obs': Eps_obs}

        new_df = pd.DataFrame(temp)
        new_df.to_csv(path,mode='a', header=False, index=False)

        # book = load_workbook(path)
        # writer = pd.ExcelWriter(path, engine='openpyxl')
        # writer.book = book
        # Eps_pairwise_df = pd.read_excel(path)
        # updated_data = pd.concat([Eps_pairwise_df,new_df],ignore_index=True)
        # updated_data.to_excel(writer, index = False)
        print(f'Mutant {genotypes} completed')
    
    print('mutants complete')
    return

def Generate_Epshat_all_TM_0(prior_mutant:None):
    '''Create large dataframe of all Epistasis at low, medium and high inducer concs'''
    TM_names = TM_stripes['genotype'].tolist()
    TM_names = list(set(TM_names[3:]))
    TM_names.sort()
    
    if prior_mutant == None:
        Eps_triplet_df = pd.DataFrame({'Genotype': [], 'Category':[], 'Epistasis_hat':[], 'Epistasis_obs': []})
        mutant_range:slice=slice(0,len(TM_names))
        Eps_triplet_df.to_csv('../results/Triplet_Eps_0.csv', index = False)
        path = f"../results/Triplet_Eps_0.csv"
    else:
        position = TM_names.index(prior_mutant)
        path = f"../results/Triplet_Eps_0.csv"
        mutant_range:slice=slice(position+1,len(TM_names))

    for genotypes in TM_names[mutant_range]:
        #get genotypeID
        mutant_list = get_mut_names(genotypes)
        #Obtain epistasis values for predictions and LAD from LMH data 
        Eps_df, G_log_mean= Generate_Eps_hat_0(mutant_list)
        
        G = G_lab(mutant_list)
        G_obs = G[0] #observed fluoresence
        low_Epsilon =  G_obs[0] - G_log_mean[0] #G_log_mean from the LAD model
        med_Epsilon =  G_obs[1] - G_log_mean[1]
        high_Epsilon =  G_obs[2] - G_log_mean[2]

        Eps_hat = [] 
        Categories = []
        Eps_obs = []

        for items in Eps_df['Eps_low'].tolist():
            Eps_hat.append(items)
            Categories.append('low')
            Eps_obs.append(low_Epsilon)
        for items in Eps_df['Eps_medium'].tolist():
            Eps_hat.append(items)
            Categories.append('medium')
            Eps_obs.append(med_Epsilon)
        for items in Eps_df['Eps_high'].tolist():
            Eps_hat.append(items)
            Categories.append('high')
            Eps_obs.append(high_Epsilon)

        Genotype_list = Eps_df['Genotype'].tolist() * 3


        temp = {'Genotype': Genotype_list, 'Category':Categories, 'Epistasis_hat':Eps_hat, 'Epistasis_obs': Eps_obs}

        new_df = pd.DataFrame(temp)
        new_df.to_csv(path,mode='a', header=False, index=False)

        # book = load_workbook(path)
        # writer = pd.ExcelWriter(path, engine='openpyxl')
        # writer.book = book
        # Eps_pairwise_df = pd.read_excel(path)
        # updated_data = pd.concat([Eps_pairwise_df,new_df],ignore_index=True)
        # updated_data.to_excel(writer, index = False)
        print(f'Mutant {genotypes} completed')
    
    print('mutants complete')
    return
    
'''split it into edible files as csv is too large'''

# Path to the large CSV file
# large_csv_path = '../results/Triplet_Eps_0.csv'

# # Number of smaller files to create
# num_files = 1000

# # Read the large CSV file in chunks
# chunk_size = 30000  # Set the chunk size according to your memory constraints
# df_chunks = pd.read_csv(large_csv_path, chunksize=chunk_size)

# #Split the data into smaller chunks
# file_counter = 1
# for chunk in df_chunks:
#     # Create a new file for each chunk
#     name = chunk['Genotype'].iloc[0]
#     smaller_csv_path = f'../results/All_triplet_data_0/mutant_{name}.csv'
#     chunk.to_csv(smaller_csv_path, index=False)
#     file_counter += 1
#     if file_counter > num_files:
#         break

def Eps_hat_to_mode(prior_mutant):
    '''Takes the output from pairwise_Eps csv's and calculates the mode, storing it in a new dataframe'''
    DM_names = DM_stripes['genotype'].tolist()
    DM_names = list(set(DM_names[3:]))
    DM_names.sort()
    count = 0
    if prior_mutant == None:
        Eps_mode_df = pd.DataFrame({'Genotype': [], 'low':[],'medium': [], 'high':[],'obs_low': [], 'obs_medium':[], 'obs_high': []})
        path = '../results/Pairwise_Mode_Eps_0.csv'
        mutant_range:slice=slice(0,len(DM_names))
        Eps_mode_df.to_csv(path, index = False)

        for i, genotypes in enumerate(DM_names[mutant_range]):
            Low_list, Medium_list, High_list, Genotype_list, Obs_Low_list, Obs_Medium_list, Obs_High_list = [], [], [], [], [], [], []

            Genotype_df = pd.read_csv(f"../results/All_pairwise_data_0/mutant_{genotypes}.csv")

            # Genotype_df = df[df.Genotype.isin([f"{genotypes}"])]

            Cato = 'Category'
            Eps_h = 'Epistasis_hat'
            Eps_o = 'Epistasis_obs'

            grouped = Genotype_df.groupby([Cato], sort = False)

            LMH = {'low': [], 'medium':[], 'high': []}
            LMH_df = pd.DataFrame(LMH)
            temp = {'obs_low': [], 'obs_medium':[], 'obs_high': []}
            temp_df = pd.DataFrame(temp)

            for j, (name, subdf) in enumerate(grouped):
                LMH_df[name] = subdf[Eps_h].tolist()
                temp_df[f'obs_{name}'] = subdf[Eps_o].tolist()
            
            l,m,h = Kde_mode(LMH_df)

            Low_list.append(l)
            Medium_list.append(m)
            High_list.append(h)
            Genotype_list.append(Genotype_df['Genotype'][0])
            Obs_Low_list.append(temp_df['obs_low'][0])
            Obs_Medium_list.append(temp_df['obs_medium'][0])
            Obs_High_list.append(temp_df['obs_high'][0])
        
            new_df = pd.DataFrame({'Genotype': Genotype_list, 'low':Low_list,'medium': Medium_list, 'high':High_list,'obs_low': Obs_Low_list, 'obs_medium':Obs_Medium_list, 'obs_high': Obs_High_list})

            count += 1

            print(f'mutant {Genotype_list}', count)
            new_df.to_csv(path,mode='a', header=False, index=False)
        print(f'mutants complete')
    else:
        position = TM_names.index(prior_mutant)
        path = '../results/Pairwise_Mode_Eps_0.csv'
        mutant_range:slice=slice(position+1,len(DM_names))
        count = prior_mutant
        for i, genotypes in enumerate(DM_names[mutant_range]):
            Low_list, Medium_list, High_list, Genotype_list, Obs_Low_list, Obs_Medium_list, Obs_High_list = [], [], [], [], [], [], []

            Genotype_df = pd.read_csv(f"../results/All_pairwise_data_0/mutant_{genotypes}.csv")

            # Genotype_df = df[df.Genotype.isin([f"{genotypes}"])]

            Cato = 'Category'
            Eps_h = 'Epistasis_hat'
            Eps_o = 'Epistasis_obs'

            grouped = Genotype_df.groupby([Cato], sort = False)

            LMH = {'low': [], 'medium':[], 'high': []}
            LMH_df = pd.DataFrame(LMH)
            temp = {'obs_low': [], 'obs_medium':[], 'obs_high': []}
            temp_df = pd.DataFrame(temp)

            for j, (name, subdf) in enumerate(grouped):
                LMH_df[name] = subdf[Eps_h].tolist()
                temp_df[f'obs_{name}'] = subdf[Eps_o].tolist()
            
            l,m,h = Kde_mode(LMH_df)

            Low_list.append(l)
            Medium_list.append(m)
            High_list.append(h)
            Genotype_list.append(Genotype_df['Genotype'][0])
            Obs_Low_list.append(temp_df['obs_low'][0])
            Obs_Medium_list.append(temp_df['obs_medium'][0])
            Obs_High_list.append(temp_df['obs_high'][0])
        
            new_df = pd.DataFrame({'Genotype': Genotype_list, 'low':Low_list,'medium': Medium_list, 'high':High_list,'obs_low': Obs_Low_list, 'obs_medium':Obs_Medium_list, 'obs_high': Obs_High_list})
            count +=1
            print(f'mutant {Genotype_list}')
            print(count)
            new_df.to_csv(path,mode='a', header=False, index=False)
        print(f'mutants complete')

def Eps_hat_to_mode_TM(prior_mutant):
    '''Takes the output from pairwise_Eps csv and calculates the mode, storing it in a new dataframe'''
    TM_names = TM_stripes['genotype'].tolist()
    TM_names = list(set(TM_names[3:]))
    TM_names.sort()
    
    if prior_mutant == None:
        Eps_mode_df = pd.DataFrame({'Genotype': [], 'low':[],'medium': [], 'high':[],'obs_low': [], 'obs_medium':[], 'obs_high': []})
        path = '../results/Triplet_Mode_Eps_0.csv'
        mutant_range:slice=slice(0,len(TM_names))
        Eps_mode_df.to_csv(path, index = False)
        count = 0
    else:
        position = TM_names.index(prior_mutant)
        path = '../results/Triplet_Mode_Eps_0.csv'
        mutant_range:slice=slice(position+1,len(TM_names))
        count = prior_mutant

    for i, genotypes in enumerate(TM_names[mutant_range]):
        Low_list, Medium_list, High_list, Genotype_list, Obs_Low_list, Obs_Medium_list, Obs_High_list = [], [], [], [], [], [], []

        Genotype_df = pd.read_csv(f"../results/All_triplet_data_0/mutant_{genotypes}.csv")

        # Genotype_df = df[df.Genotype.isin([f"{genotypes}"])]

        Cato = 'Category'
        Eps_h = 'Epistasis_hat'
        Eps_o = 'Epistasis_obs'

        grouped = Genotype_df.groupby([Cato], sort = False)

        LMH = {'low': [], 'medium':[], 'high': []}
        LMH_df = pd.DataFrame(LMH)
        temp = {'obs_low': [], 'obs_medium':[], 'obs_high': []}
        temp_df = pd.DataFrame(temp)

        for j, (name, subdf) in enumerate(grouped):
            LMH_df[name] = subdf[Eps_h].tolist()
            temp_df[f'obs_{name}'] = subdf[Eps_o].tolist()
        
        l,m,h = Kde_mode(LMH_df)

        Low_list.append(l)
        Medium_list.append(m)
        High_list.append(h)
        Genotype_list.append(Genotype_df['Genotype'][0])
        Obs_Low_list.append(temp_df['obs_low'][0])
        Obs_Medium_list.append(temp_df['obs_medium'][0])
        Obs_High_list.append(temp_df['obs_high'][0])
    
        new_df = pd.DataFrame({'Genotype': Genotype_list, 'low':Low_list,'medium': Medium_list, 'high':High_list,'obs_low': Obs_Low_list, 'obs_medium':Obs_Medium_list, 'obs_high': Obs_High_list})

        count += 1

        print(f'mutant {Genotype_list}', count)
        new_df.to_csv(path,mode='a', header=False, index=False)
    print(f'mutants complete')
        
#test to visualise low epistasis distribtion
def Eps_distribution(mutant_list:list):
    '''Visualises the distribution of epistasis'''
    path = f"../results/Pairwise_Eps.xlsx"
    df = pd.read_excel(path)
    genotypes = get_mut_ids(mutant_list)

    Genotype_df = df[df.Genotype.isin([f"{genotypes}"])]

    Cato = 'Category'
    Eps_h = 'Epistasis_hat'
    Eps_o = 'Epistasis_obs'

    grouped = Genotype_df.groupby([Cato], sort = False)

    LMH = {'low': [], 'medium':[], 'high': []}
    LMH_df = pd.DataFrame(LMH)
    Obs = {'obs_low': [], 'obs_medium':[], 'obs_high': []}
    Obs_df = pd.DataFrame(Obs)

    for i, (name, subdf) in enumerate(grouped):
        LMH_df[name] = subdf[Eps_h].tolist()
        Obs_df[f'obs_{name}'] = subdf[Eps_o].tolist()
    
    low = LMH_df['low'].tolist()
    med = LMH_df['medium'].tolist()
    high = LMH_df['low'].tolist()

    low_kde = gaussian_kde(low)
    medium_kde = gaussian_kde(med)
    high_kde = gaussian_kde(high)

    x = np.linspace(min(low),max(low),num=1000)
    y = low_kde(x)

    mode_index = np.argmax(y)
    mode_x = x[mode_index]
    Obs_mode_x = Obs_df['obs_low'][0]

    plt.hist(x, y, color='salmon', label='Distribution of Epistasis')
    plt.xlabel('Predicted Epistasis hat')
    plt.ylabel('Density')
    plt.title(f'Distribution of Epistasis at low Iconc for mutant {genotypes}')
    plt.axvline(mode_x, color='salmon', linestyle='--', label='Mode Epistasis')
    plt.axvline(Obs_mode_x, color='darkcyan', linestyle='--', label='Observed Epistasis')
    plt.legend()

    plt.show()
import matplotlib.ticker as mticker

#Visualise all distribution across inducer concentrations for pairwise or triplet.
def Eps_hist(pair:bool, save:bool):
    '''Takes a list of mutants to look at the distribution of epistasis values, compares to observed distribution'''
    if pair == True:
        DM_names = DM_stripes['genotype'].tolist()
        DM_names = list(set(DM_names[3:]))
        DM_names.sort()
        path = '../results/Pairwise_Mode_Eps.csv'
        names = DM_names
        set_name = 'pairwise'
    if pair == False:
        TM_names = TM_stripes['genotype'].tolist()
        TM_names = list(set(TM_names[3:]))
        TM_names.sort()
        path = '../results/Triplet_Mode_Eps.csv'
        names = TM_names
        set_name = 'Triplet'
    df = pd.read_csv(path)

    All_Eps_hat = []
    All_Eps_obs = []
    mutant_range:slice=slice(0,len(names))
    for genotypes in names[mutant_range]:
        #Get all Eps values in observed data
        
        subdf = df[df['Genotype'] == genotypes]
        All_Eps_obs.append(subdf['obs_low'].iloc[0])
        All_Eps_obs.append(subdf['obs_medium'].iloc[0])
        All_Eps_obs.append(subdf['obs_high'].iloc[0])
    

        #Get all Eps

        subdf = df[df['Genotype'] == genotypes]
        All_Eps_hat.append(subdf['low'].iloc[0])
        All_Eps_hat.append(subdf['medium'].iloc[0])
        All_Eps_hat.append(subdf['high'].iloc[0])


    fig = plt.figure()

    plt.hist(All_Eps_hat, edgecolor = 'salmon',label = 'Model', bins='auto', linewidth=1, density=True, fill = False, histtype="stepfilled")
    plt.grid(visible=False)
    plt.hist(All_Eps_obs, edgecolor = 'darkcyan', label = 'Observed',bins='auto', linewidth=1, density=True, fill = False, histtype="stepfilled")
    plt.grid(visible=False)
    plt.xlabel('Epistasis')
    plt.ylabel('density')
    plt.title(f'Distribution of all {set_name} Epistasis values')
    meanhat = np.mean(All_Eps_hat)
    meanobs = np.mean(All_Eps_obs)
    stdhat = np.std(All_Eps_hat)
    stdobs = np.std(All_Eps_obs)

    
    plt.axvline(meanhat, color='salmon', linestyle='--', label='Mean Model')
    plt.axvline(meanobs, color='darkcyan', linestyle='--', label='Mean Observed') 
    plt.legend()
    txt = f'Mean and Stdev of model epistasis: \n'
    txt+=str(meanhat)
    txt+= '\n'
    txt+= str(stdhat)
    txt2 = f'Mean and Stdev of observed epistasis: \n'
    txt2+=str(meanobs)
    txt2+= '\n'
    txt2+= str(stdobs)
    fig.text(0,0,txt,wrap=True, fontsize=6)
    fig.text(0,-.1,txt2,wrap=True, fontsize=6)
    if save == True:
        plt.savefig(f'../results/Eps_all_{set_name}_hist.pdf', format="pdf", bbox_inches="tight")
    plt.show
#Combined epistasis of all combined mutants.
def Eps_hist_all(save:bool):
    '''Takes a list of mutants to look at the distribution of epistasis values, compares to observed distribution'''
    #Change to TM_names, df2 if triplet.

    DM_names = DM_stripes['genotype'].tolist()
    DM_names = list(set(DM_names[3:]))
    DM_names.sort()
    path1 = '../results/Pairwise_Mode_Eps.csv'
    path0 = '../results/Pairwise_Mode_Eps_0.csv' #centralised mean=0
    TM_names = TM_stripes['genotype'].tolist()
    TM_names = list(set(TM_names[3:]))
    TM_names.sort()
    path2 = '../results/Triplet_Mode_Eps.csv'
    
    
    df1 = pd.read_csv(path1)
    df0 = pd.read_csv(path0) #centralised mean = 0
    df2 = pd.read_csv(path2)

    All_Eps_hat = []
    All_Eps_obs = []
    All_Eps_hat_0 = []

    mutant_range:slice=slice(0,len(DM_names))
    for genotypes in DM_names[mutant_range]:
        #Get all Eps values in observed data
        
        subdf = df1[df1['Genotype'] == genotypes]
        All_Eps_obs.append(subdf['obs_low'].iloc[0])
        All_Eps_obs.append(subdf['obs_medium'].iloc[0])
        All_Eps_obs.append(subdf['obs_high'].iloc[0])
    

        #Get all Eps
        subdf = df1[df1['Genotype'] == genotypes]
        All_Eps_hat.append(subdf['low'].iloc[0])
        All_Eps_hat.append(subdf['medium'].iloc[0])
        All_Eps_hat.append(subdf['high'].iloc[0])

        #Visualising mean=0 distribution
        # subdf = df0[df0['Genotype'] == genotypes]
        # All_Eps_hat_0.append(subdf['low'].iloc[0])
        # All_Eps_hat_0.append(subdf['medium'].iloc[0])
        # All_Eps_hat_0.append(subdf['high'].iloc[0])


    fig = plt.figure()

    plt.hist(All_Eps_hat, edgecolor = 'salmon',label = 'Model', bins='auto', linewidth=1, density=True, fill = False, histtype="stepfilled")
    plt.grid(visible=False)
    # plt.hist(All_Eps_hat_0, edgecolor = 'indigo',label = 'Model_0', bins='auto', linewidth=1, density=True, fill = False, histtype="stepfilled")
    # plt.grid(visible=False) # centralised mean = 0
    plt.hist(All_Eps_obs, edgecolor = 'darkcyan', label = 'Observed',bins='auto', linewidth=1, density=True, fill = False, histtype="stepfilled")
    plt.grid(visible=False)
    plt.xlabel('Epistasis')
    plt.ylabel('density')
    plt.title('Distribution of Triplet Epistasis values')
    
    meanhat = np.mean(All_Eps_hat)
    meanobs = np.mean(All_Eps_obs)
    meanhat0 = np.mean(All_Eps_hat_0)
    stdhat = np.std(All_Eps_hat)
    stdobs = np.std(All_Eps_obs)

    
    plt.axvline(meanhat, color='salmon', linestyle='--', label='Mean Model')
    plt.axvline(meanobs, color='darkcyan', linestyle='--', label='Mean Observed') 
    #plt.axvline(meanhat0, color='indigo', linestyle='--', label='Mean Model0')
    plt.legend()
    txt = f'Mean and Stdev of model epistasis: \n'
    txt+=str(meanhat)
    txt+= '\n'
    txt+= str(stdhat)
    txt2 = f'Mean and Stdev of observed epistasis: \n'
    txt2+=str(meanobs)
    txt2+= '\n'
    txt2+= str(stdobs)
    fig.text(0,0,txt,wrap=True, fontsize=6)
    fig.text(0,-.1,txt2,wrap=True, fontsize=6)
    if save == True:
        plt.savefig(f'../results/Triplet_Eps_all_hist_0.pdf', format="pdf", bbox_inches="tight")
    plt.show
    print(meanhat0)

#LMH visualisation of epistasis
def LMH_hist_all(save:bool):
    ''' distribution of epistasis values at LMH, compares to observed distribution'''
    
    DM_names = DM_stripes['genotype'].tolist()
    DM_names = list(set(DM_names[3:]))
    DM_names.sort()
    path1 = '../results/Pairwise_Mode_Eps.csv'
    TM_names = TM_stripes['genotype'].tolist()
    TM_names = list(set(TM_names[3:]))
    TM_names.sort()
    path2 = '../results/Triplet_Mode_Eps.csv'
    
    
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    # combined_df = pd.concat([df1,df2], ignore_index=True)
    split_columns = [('low', 'obs_low'),('medium','obs_medium'), ('high','obs_high')]

    fig, axes = plt.subplots(nrows=1, ncols=len(split_columns), figsize=(12, 4))

    for i, (col1, col2) in enumerate(split_columns):
        data=pd.melt(df2[[col1, col2]])
        data[f'{col1}'] = 0 + i
        sns.violinplot(data=pd.melt(df2[[col1, col2]]), x=data[f'{col1}'], y='value', hue='variable', split=True, palette = ['salmon','darkcyan'], ax=axes[i], fill=None)
        axes[i].set_title(f'{col1} and {col2}')
        axes[i].legend().remove()
        axes[i].set_ylabel('')
    axes[0].set_ylabel('Epistasis')
    fig.suptitle('Triplet')
    Wildtype = mpatches.Patch(color= 'darkcyan', label='Wildtype') 
    data_set = mpatches.Patch(color= 'salmon', label='triplet data')
    plt.legend(handles=[data_set,Wildtype], bbox_to_anchor=(1, 1), title = "Legend")

    # sns.violinplot(df1_low, split=True)
    if save == True:
        plt.savefig(f'../results/Triplet_LMH.pdf', format="pdf", bbox_inches="tight")
    plt.show

#%%
'''Looking at the distribution of modifier parameters'''
def Mods_Hist():
    '''Manually change the node and parameter to look at different distributions.'''
    node = "Regulator"
    all_O = []
    for i in range(1,11):
        path = f'../data/smc_SM_hill/{node}{i}_smc/pars_final.out'
        df = Out_to_DF_hill(path, model_hill.model_muts, node, all=False) #named
        temp = df['MAr'].to_list()
        all_O.extend(temp)
        # palette = [
        #     "#FFA07A",
        #     "#FF8C69",
        #     "#FF7F50",
        #     "#FF7256",
        #     "#FF6347",
        #     "#FF4500",
        #     "#FF3C00",
        #     "#FF3200",
        #     "#FF2800",
        #     "#FF1E00"
        # ]
        palette = [
            "#333333",
            "#3C3C3C",
            "#454545",
            "#4E4E4E",
            "#575757",
            "#606060",
            "#696969",
            "#737373",
            "#7C7C7C",
            "#858585"
        ]
        # palette = [
        #     "#008B8B",
        #     "#008080",
        #     "#007D7D",
        #     "#007676",
        #     "#007070",
        #     "#006969",
        #     "#006666",
        #     "#006060",
        #     "#005959",
        #     "#005353"
        # ]
        plt.hist(temp,label = f'R{i}',edgecolor = palette[i-1], bins='auto', linewidth=1, density=True, fill = False, histtype="stepfilled", alpha = 0.2)
    plt.hist(all_O, edgecolor = 'darkgrey',label = 'Regulator', bins='auto', linewidth=3, density=True, fill = False, histtype="stepfilled")
    plt.title('Regulator mutants, distribution of MAr')
    plt.xlabel('Parameter Values (log)')
    plt.ylabel('Density/Credibility')
    plt.legend()






        
        
        


        
        










    
    
   


        
     



         




#%%
