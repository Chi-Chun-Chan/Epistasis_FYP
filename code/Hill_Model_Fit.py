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

param_dist = multivariate_dis(WT_converged_params, 13)

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
def Visualise_SM_fit(mut_name, iter, plot_num):
    '''Looking at the general fits to data'''
    path = f'../data/smc_SM_hill/{mut_name}_smc/all_pars_{iter}.out'  #iter = final
    path2 = f'../data/smc_SM_hill/{mut_name}_smc/pars_{iter}.out'  #only modifiers
    df = Out_to_DF_hill(path, model_hill.model_muts, mut_name, all=True)

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


fig, scores = Visualise_SM_fit(mut_name='Regulator1', iter = 'final', plot_num = 50)
print(scores)

def Visualise_SM_par_dis(mut_name, iter):
    path2 = f'../data/smc_SM_hill/{mut_name}_smc/pars_{iter}.out'  #only modifiers
    df2 = Out_to_DF_hill(path2, model_hill.model_muts, mut_name, all=False)
    
    return Paired_Density_plot_mut(df2, name = mut_name, save = True)

Visualise_SM_par_dis(mut_name='Regulator1', iter = 'final')

# %%
'''Visualisation of pairwise/triplet fits using combined modifiers'''
import time
def Visualise_combo_mut_fit(mutants:list):
    '''Takes 2 or 3 mutants in a list and plots them to the data, enter mutants as follows [Output/Sensor/Regulator[1-10]]'''

    random_params = param_dist.rvs(size=1, random_state=rndint+timeseed)
    if len(mutants) == 2:
        mut1 = mutants[0]
        mut2 = mutants[1]

        rndint = np.random.randint(low=0, high=1e7)
        timeseed = time.time_ns() % 2**16
        np.random.seed(rndint+timeseed)
        seed(rndint+timeseed)

        #WT params
        path = '../data/smc_hill/pars_final.out'
        WT_converged_params = Out_to_DF_hill(path, model_hill, mut_name= "", all = False)
        param_dist = multivariate_dis(WT_converged_params,13)
        WT_pars_array = param_dist.rvs(size=50, random_state=rndint+timeseed)

        #mutant1 modifiers
        path = f'../data/smc_SM_hill/{mut1}_smc/pars_final.out'  #only modifiers
        df1 = Out_to_DF_hill(path, model_hill.model_muts, mut1, all=False)
        MD1 = multivariate_dis(df1)
        mut1_pars_array = MD1.rvs(size=50, random_state=rndint+timeseed)

        #mutant2 modifiers
        path2 = f'../data/smc_SM_hill/{mut2}_smc/pars_final.out'  #only modifiers
        df2 = Out_to_DF_hill(path2, model_hill.model_muts, mut2, all=False)
        MD2 = multivariate_dis(df2)
        mut2_pars_array = MD2.rvs(size=50, random_state=rndint+timeseed)

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
        

        
        pairwise = f'{pair1}_{pair2}'
        #need to make more ifelse statements to reorganise pairwise name.

        #need to access meta_dict to select low, med and high values

        # alpha = []
        # for items in meta_dict['DM']:
        #     if items['genotype'] == 'R1_O1' & items['inducer level'] == 'low':
        #     alpha.append(items['obs_fluo_mean'])
        # alpha




        #plot mutant fits
        hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)
        for WT_pars,Mut1_pars,Mut2_pars in zip(WT_pars_array,mut1_pars_array,mut2_pars_array):
            #identification of mutant types
            if mut1.startswith('Sensor') & mut2.startswith('Output'):
                M = {'As':Mut1_pars[0],'Bs':Mut1_pars[1],'Cs':Mut1_pars[2],'Ns':Mut1_pars[3],'Ar':0.0,'Br':0.0,'Cr':0.0,'Nr':0.0,'Ao':Mut2_pars[0],'Bo':Mut2_pars[1],'Co':Mut2_pars[2],'No':Mut2_pars[3],'Fo':Mut2_pars[4]}
            elif mut1.startswith('Sensor') & mut2.startswith('Regulator'):
                M = {'As':Mut1_pars[0],'Bs':Mut1_pars[1],'Cs':Mut1_pars[2],'Ns':Mut1_pars[3],'Ar':Mut2_pars[0],'Br':Mut2_pars[1],'Cr':Mut2_pars[2],'Nr':Mut2_pars[3],'Ao':0.0,'Bo':0.0,'Co':0.0,'No':0.0,'Fo':0.0}
            elif mut1.startswith('Regulator') & mut2.startswith('Output'):
                M = {'As':0.0,'Bs':0.0,'Cs':0.0,'Ns':0.0,'Ar':Mut1_pars[0],'Br':Mut1_pars[1],'Cr':Mut1_pars[2],'Nr':Mut1_pars[3],'Ao':Mut2_pars[0],'Bo':Mut2_pars[1],'Co':Mut2_pars[2],'No':Mut2_pars[3],'Fo':Mut2_pars[4]}
            elif mut1.startswith('Regulator') & mut2.startswith('Sensor'):
                M = {'As':Mut2_pars[0],'Bs':Mut2_pars[1],'Cs':Mut2_pars[2],'Ns':Mut2_pars[3],'Ar':Mut1_pars[1],'Br':Mut1_pars[1],'Cr':Mut1_pars[1],'Nr':Mut1_pars[1],'Ao':0.0,'Bo':0.0,'Co':0.0,'No':0.0,'Fo':0.0}
            elif mut1.startswith('Output') & mut2.startswith('Regulator'):
                M = {'As':0.0,'Bs':0.0,'Cs':0.0,'Ns':0.0,'Ar':Mut2_pars[0],'Br':Mut2_pars[1],'Cr':Mut2_pars[2],'Nr':Mut2_pars[3],'Ao':Mut1_pars[0],'Bo':Mut1_pars[1],'Co':Mut1_pars[2],'No':Mut1_pars[3],'Fo':Mut1_pars[4]}
            elif mut1.startswith('Output') & mut2.startswith('Sensor'):
                M = {'As':Mut2_pars[0],'Bs':Mut2_pars[1],'Cs':Mut2_pars[2],'Ns':Mut2_pars[3],'Ar':0.0,'Br':0.0,'Cr':0.0,'Nr':0.0,'Ao':Mut1_pars[0],'Bo':Mut1_pars[1],'Co':Mut1_pars[2],'No':Mut1_pars[3],'Fo':Mut1_pars[4]}
            else:
                raise KeyError('Mutant names invalid 212')
            par_dict = {
                "A_s":WT_pars[0],
                "B_s":WT_pars[1],
                "C_s":WT_pars[2],
                "N_s":WT_pars[3],
                "MA_s":M['As'],
                "MB_s":M['Bs'],
                "MC_s":M['Cs'],
                "MN_s":M[4], #changed to log
                "A_r":WT_pars[4],
                "B_r":WT_pars[5],
                "C_r":WT_pars[6],
                "N_r":WT_pars[7],
                "MA_r":M['Ar'],
                "MB_r":M['Br'],
                "MC_r":M['Cr'],
                "MN_r":M['Nr'],
                "A_o":WT_pars[8],
                "B_o":WT_pars[9],
                "C_o":WT_pars[10],
                "N_o":WT_pars[11],
                "F_o":WT_pars[12],
                "MA_o":M['Ao'],
                "MB_o":M['Bo'],
                "MC_o":M['Co'],
                "MN_o":M['No'],
                "MF_o":M['Fo'],
                    }
            
            par_list = list(par_dict.values())

            Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model_muts(I_conc=data_.S,params_list=par_list)



            



    elif len(mutants) == 3:
        mut1 = mutants[0]
        mut2 = mutants[1]
        mut3 = mutants[2]
    else:
        raise KeyError('incorrect number of mutants, must be 2 or 3')



# %%
