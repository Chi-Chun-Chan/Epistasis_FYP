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
# dataframe = pd.read_csv('../data/smc/pars_final.out')
# dataf = pd.read_csv(dataframe, delimiter=' ', index_col=None, header=None)

'''Visualising Hill WT parameter distributions'''
start_time_per_mutant=time.time()

path = '../data/smc_hill/pars_final.out'

WT_converged_params = Out_to_DF_hill(path, model_hill.model, '', all = False)

param_dist = multivariate_dis(WT_converged_params, 13)

# np.random.seed(0)  
# rndint = np.random.randint(low=0, high=1e7)
    
# timeseed = time.time_ns() % 2**16
# np.random.seed(rndint+timeseed)
# seed(rndint+timeseed)

# random_params = param_dist.rvs(size=1, random_state=rndint+timeseed)


#plot the parameters of As and Bs as scatterplot
#generate 100 random parameters sets and then plot that on top of the scatter to visualise whether
#they fit in the distribution.

#test to see if sampling from multi-variate distribution works
# A_s = []
# A_s = WT_converged_params['Br'].to_numpy()

# B_s = []
# B_s = WT_converged_params['Cr'].to_numpy()

# A_s_gaus = []
# B_s_gaus = []
# temp = param_dist.rvs(size=100, random_state=rndint+timeseed+200)

# for items in temp:
#     A_s_gaus.append(items[5])
#     B_s_gaus.append(items[6])

# plt.scatter(A_s,B_s, c = 'b')
# plt.scatter(A_s_gaus,B_s_gaus, c ='r') 




#sample_prior()

#for each sample prior, select random_params using a random seed
#then use each one e.g random_params[0] as the prior for that variable
#Still use three different sample_prior functions. Still need three par_lists, then Modifier element for mutant params.

# par_list = list(df.columns)
# df = pd.melt(df, value_vars=par_list, var_name='params', value_name='Param value',ignore_index=False)

WT = 'Wildtype'
Paired_Density_plot(WT_converged_params, name = WT)




#%%
Hill_model = model_hill(params_list=[1]*13, I_conc=meta_dict["WT"].S)
func = Hill_model.model

# params_hill_dict={"sen_params":{"A_s":767.1584089405626,"B_s":16942.01930176865,"C_s":896.97,"N_s":1.151181955178552},"reg_params":{"A_r":2229.803862083969,"B_r":8961.65164532133,"C_r":0.001461383502353,"N_r":1.84
# },"out_h_params":{},"out_params":{"A_o":985.9836597373027,"B_o":18015297.65499306,"C_o":0.101052174093944,"N_o":1.417995609958926},"free_params":{"F_o":1.477610561220013}}#RSS 0.161
# params_hill_dict={"sen_params":{"A_s":10**2.838248875556981687e+00,"B_s":10**4.295428708881752655e+00,"C_s":10**2.779099379130348879e+00,"N_s":1.020179391796438129e+00},"reg_params":{"A_r":10**3.308479807112040927e+00,"B_r":10**3.641716161424724429e+00,"C_r":10**-3.110782482589740994e+00,"N_r":1.786350481190633221e+00
# },"out_h_params":{},"out_params":{"A_o":10**3.080940458532804183e+00,"B_o":10**6.789192470034910443e+00,"C_o":10**-1.165526383628718854e+00,"N_o":1.309353801416837548e+00},"free_params":{"F_o":1.0e+00}}
# params_hill_list=dict_to_list(params_hill_dict)

params_hill_dict={"sen_params":{"A_s":10**2.881002710475187190e+00,"B_s":10**4.234114461927148021e+00,"C_s":10**2.917517255582483315e+00,"N_s":1.125732163978979461e+00},"reg_params":{"A_r":10**3.367414514911509116e+00,"B_r":10**3.841889231200538379e+00,"C_r":10**-2.898387275917982286e+00,"N_r":1.742140699208641008e+00
},"out_h_params":{},"out_params":{"A_o":10**3.039548305373660053e+00,"B_o":10**4.894183658414895888e+00,"C_o":10**-2.822117047581649274e+00,"N_o":1.688399702374762335e+00},"free_params":{"F_o":1.522445716338543864e+00}}
params_hill_list=dict_to_list(params_hill_dict)

converged_params_list_hill=Plotter(model_type=func,start_guess=params_hill_list,n_iter=1e5,method="Nelder-Mead",params_dict=params_hill_dict,custom_settings=[],tol=0.0001,mutation='Wildtype')

data = meta_dict["WT"]
RSS_Score(params_hill_list,model_hill,data, model_specs='None')
# %%
converged_params_list_hill=get_WT_params(model_type=func,start_guess=params_hill_list,n_iter=1e5,method="Nelder-Mead",params_dict=params_hill_dict,custom_settings=[],tol=0.0001)
#get_WT_params(model_type=func,start_guess=params_hill_list,n_iter=1e5,method="Nelder-Mead",params_dict=params_hill_dict,custom_settings=[],tol=0.0001)
#plt.savefig('../results/Hill_WT_Fit.pdf', format="pdf", bbox_inches="tight")
# %%
#testing abc_smc for SM
Hill_model = model_hill(params_list=[1]*13, I_conc=meta_dict["WT"].S)
func = Hill_model.model
d = get_data_SM("Output2")

params_dict = {"sen_params":{"A_s":10**2.881002710475187190e+00,"B_s":10**4.234114461927148021e+00,"C_s":10**2.917517255582483315e+00,"N_s":1.125732163978979461e+00},"reg_params":{"A_r":10**3.367414514911509116e+00,"B_r":10**3.841889231200538379e+00,"C_r":10**-2.898387275917982286e+00,"N_r":1.742140699208641008e+00
},"out_h_params":{},"out_params":{"A_o":10**3.640658721691746535e+00,"B_o":10**4.998187132976670277e+00,"C_o":10**-2.646560931451678922e+00,"N_o":1.143877351914554641e+00},"free_params":{"F_o":9.577592265972070251e-01}}

params_list=dict_to_list(params_dict)

Plotter(model_type=func,start_guess=params_list,n_iter=1e5,method="Nelder-Mead",params_dict=params_dict,custom_settings=[],tol=0.0001,mutation='Output2')
RSS_Score(params_list,model_hill,d, model_specs='None')



# %%
'''Plotting functions to visualise Hill params'''
#Visualising parameter distribution
from Plotting_functions import *
from RSS_Scoring import *
def Visualise_SM_par(mut_name, iter, plot_num):
    '''Looking at the parameter distribution and fits'''
    path = f'../data/smc_SM_hill/{mut_name}_smc/all_pars_{iter}.out'  #iter = final
    path2 = f'../data/smc_SM_hill/{mut_name}_smc/pars_{iter}.out'  #only modifiers
    df = Out_to_DF_hill(path, model_hill.model_muts, mut_name, all=True)
    # df2 = Out_to_DF_hill(path2, model_hill.model_muts, mut_name, all=False)
    # Paired_Density_plot_mut(df2, name = mut_name)

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
    par_array = np.empty([plot_num,26])
    for i in range(1,plot_num+1):
        row_list = df.loc[i].values.flatten().tolist()
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

        score_list.append(RSS_Score(par_array[i],model_hill,data_,model_specs='model_muts'))

    # Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model_muts(I_conc=data_.S,params_list=mean_params)
    # Stripe.plot(data_.S, Stripe_est_array, alpha = 0.1, c = 'black')
    # Output.plot(data_.S, Output_est_array, alpha = 0.1, c = 'black')
    # Regulator.plot(data_.S, Regulator_est_array,alpha = 0.1, c = 'black')
    # Sensor.plot(data_.S, Sensor_est_array, alpha = 0.1, c = 'black')
    # return fig, score_list, par_array

#need to figure out whats wrong with model muts and why its producing dog shit
    fig.suptitle(f'{mut_name} Mutant Fitting with {plot_num} parameter sets')
    plt.savefig(f'../results/{mut_name}_SM_fit.pdf', format="pdf", bbox_inches="tight")

    return fig, score_list


fig, scores = Visualise_SM_par(mut_name='Regulator1', iter = 'final', plot_num = 50)
print(scores)


    








# %%
from Plotting_functions import *
mut_name = 'Regulator1'
iter = 'final'
path2 = f'../data/smc_SM_hill/{mut_name}_smc/pars_{iter}.out'  #only modifiers
df2 = Out_to_DF_hill(path2, model_hill.model_muts, mut_name, all=False)
Paired_Density_plot_mut(df2, name = mut_name, save = True)
# %%
