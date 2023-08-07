#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from Models import *
from data_wrangling import *
from Model_fitting_functions import *
from random import seed
import time
from Plotting_functions import *
from RSS_Scoring import *
#%%
'''Visualising WT fit to data'''
hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)
path = '../data/smc_WT_new/pars_final.out'

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
par_array = np.empty([plot_num,14])
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

    #write a line to 10** all the necessary parameters before plugging into model
for pars in par_array:
    pars[0] = 10**pars[0] #A_s
    pars[1] = 10**pars[1] #B_s
    pars[2] = 10**pars[2] #C_s
    pars[4] = 10**pars[4] #A_r
    pars[5] = 10**pars[5] #B_r
    pars[6] = 10**pars[6] #C_r
    pars[8] = 10**pars[8] #A_o
    pars[9] = 10**pars[9] #B_o
    pars[10] = 10**pars[10] #C_o
    pars[11] = 10**pars[11] #C_k
    pars[13] = 10**pars[13] #f_o

for i in range(0,len(par_array)):
    Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model_new_WT(I_conc=data_.S,params_list=par_array[i])

    Stripe.plot(data_.S, Stripe_est_array, alpha = 0.1, c = 'teal')
    Output.plot(data_.S, Output_est_array, alpha = 0.1, c = 'steelblue')
    Regulator.plot(data_.S, Regulator_est_array,alpha = 0.1, c = 'dimgrey')
    Sensor.plot(data_.S, Sensor_est_array, alpha = 0.1, c = 'darkorange')
# %%
def Visualise_SM_fit(mut_name, iter, plot_num, save:bool):
    '''Looking at the general fits to data'''
    path = f'../data/smc_hill_new/{mut_name}_smc/all_pars_{iter}.out'  #iter = final
    path2 = f'../data/smc_hill_new/{mut_name}_smc/pars_{iter}.out'  #only modifiers
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

Visualise_SM_fit(mut_name='Regulator1',iter = 14, plot_num= 50, save=False)

# %%
