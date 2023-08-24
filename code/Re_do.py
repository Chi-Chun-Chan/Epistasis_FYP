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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import progressbar as pb
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
    fig.show()
    if save == True:
        plt.savefig(f'../results/{mut_name}_SM_fit.pdf', format="pdf", bbox_inches="tight")

    return #fig # score_list, set_list

#Visualise_SM_fit(mut_name='Regulator6',iter = 'final', plot_num= 50, save=False)

#%%
m1_est = WT_sample
m1_est = 10**m1_est
m1_est[4] = m1_est[4]*(10**M1_cond_params[1][0])
m1_est[5] = m1_est[5]*(10**M1_cond_params[1][1])
m1_est[6] = m1_est[6]*(10**M1_cond_params[1][2])
m1_est[7] = m1_est[7]*(10**M1_cond_params[1][3])
m2_est = WT_sample
m2_est = 10**m2_est
m2_est[8] = m2_est[8]*(10**M2_cond_params[1][0])
m2_est[9] = m2_est[9]*(10**M2_cond_params[1][1])
m2_est[10] = m2_est[10]*(10**M2_cond_params[1][2])
m2_est[12] = m2_est[12]*(10**M2_cond_params[1][3])
# hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)
# data = meta_dict['WT']
# fig, ((Sensor, Regulator), (Output, Stripe)) = plt.subplots(2,2, constrained_layout=True)
# Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model_new_WT(I_conc=data.S,params_list=10**WT_sample)
# Stripe.plot(data.S, Stripe_est_array, c = 'teal')
# Output.plot(data.S, Output_est_array, c = 'steelblue')
# Regulator.plot(data.S, Regulator_est_array, c = 'dimgrey')
# Sensor.plot(data.S, Sensor_est_array, c = 'darkorange')
# Output.set_xscale('log')
# Output.set_yscale('log')
# Sensor.set_xscale('log')
# Sensor.set_yscale('log')
# Regulator.set_xscale('log')
# Regulator.set_yscale('log')
# Stripe.set_xscale('log')
# Stripe.set_yscale('log')
# %%
'''Pairwise parameters'''

def get_sm_params(mutants:list):
    mut1 = mutants[0]
    mut2 = mutants[1]
    mut_id = get_mut_ids(mutants)
    path = f'../data/smc_hill_new/{mut1}_smc/all_pars_final.out'
    path2 = f'../data/smc_hill_new/{mut2}_smc/all_pars_final.out'
    df1 = Out_to_DF_hill(path, model_hill.model_muts, mut1, all=True)
    df2 = Out_to_DF_hill(path2, model_hill.model_muts, mut2, all=True)
    WT1_df = df1[['As','Bs','Cs','Ns','Ar','Br','Cr','Nr','Ao','Bo','Co','Ck','No','Fo']]
    WT1_df = np.log10(WT1_df) #convert to log10
    WT1_df.reset_index(drop=True, inplace=True)
    WT2_df = df2[['As','Bs','Cs','Ns','Ar','Br','Cr','Nr','Ao','Bo','Co','Ck','No','Fo']]
    WT2_df = np.log10(WT2_df) #convert to log10
    WT2_df.reset_index(drop=True, inplace=True)

    letters = r'[a-zA-Z]'

    # for file in listdir(folder):

    # Extract from file title, which is the duplet
    mutant_letters = re.findall(letters, mut_id)
    m1_str = mutant_letters[0].lower() # first character of filename string
    m2_str = mutant_letters[1].lower()

    M1_mods_df = df1[[f'MA{m1_str}',f'MB{m1_str}',f'MC{m1_str}',f'MN{m1_str}']]
    M2_mods_df = df2[[f'MA{m2_str}',f'MB{m2_str}',f'MC{m2_str}',f'MN{m2_str}']]
    M1_mods_df = np.log10(M1_mods_df)
    M2_mods_df = np.log10(M2_mods_df)

    # mod_path = f'../data/smc_SM_hill/{mut1}_smc/pars_final.out' 
    # M1_mods_df = Out_to_DF_hill(mod_path, model_hill.model_muts, mut1, all=False)
    M1_mods_df.reset_index(drop=True, inplace=True)
    # mod_path2 = f'../data/smc_SM_hill/{mut2}_smc/pars_final.out' 
    # M2_mods_df = Out_to_DF_hill(mod_path2, model_hill.model_muts, mut2, all=False)
    M2_mods_df.reset_index(drop=True, inplace=True)

    M1_df = pd.concat([M1_mods_df,WT1_df], axis=1)
    M2_df = pd.concat([M2_mods_df, WT2_df,], axis=1)
    Combined_WT = pd.concat([WT1_df,WT2_df], axis=0)

    return Combined_WT, M1_df, M2_df, M1_mods_df, M2_mods_df #returns all parameters in log10 form

def get_dm_params(mutants:list):
    data = meta_dict['WT']
    combined_wt, m1_wt, m2_wt, m1, m2 = get_sm_params(mutants) #extracts the fitted parameters and corresponding WT params
    names = combined_wt.keys()
    params = len(combined_wt.columns)
    WT_matrix = np.empty(shape=(params,2000), dtype=float)
    i = 0
    for name in names:
        WT_matrix[i] = combined_wt[name].to_numpy() #takes list of single params and appends to array.
        i = i+1

    WT_mean_list = []
    j = 0

    for m in WT_matrix: #calculates the mean of each logged parameter of WT
        means = sum(m)
        means = means/len(m)
        WT_mean_list.append(means)
        j = j+1
    #generate cov matrix
    combined_wt = combined_wt.T
    WT_cov_matrix = np.cov(combined_wt.values) #covariance of each params 14x14 matrix
    #generate multivariate normal distribution
    WT_multi_norm_dis = multivariate_normal(
                        mean = WT_mean_list,
                        cov = WT_cov_matrix,
                        allow_singular = True)
    #accurate = False
    # while accurate == False:
    #     rndint = np.random.randint(low=0, high=1e7)
    #     timeseed = time.time_ns() % 2**16
    #     np.random.seed(rndint+timeseed)
    #     seed(rndint+timeseed)
    #     WT_sample = WT_multi_norm_dis.rvs(size=1, random_state=rndint+timeseed) #generates one wildtype sample from the shared distribution
    #     distance = RSS_Score(param_list= 10**WT_sample, model_type=model_hill, data_=data, model_specs='new_WT')
    #     if distance <= 0.08: #only select good WT params
    #         accurate = True
    rndint = np.random.randint(low=0, high=1e7) #generates random timeseed
    timeseed = time.time_ns() % 2**16
    np.random.seed(rndint+timeseed)
    seed(rndint+timeseed)
    WT_sample = WT_multi_norm_dis.rvs(size=1, random_state=rndint+timeseed) #selects one WT parameter sampled from the combined distribution of params
    #############################################################
    #Sample from modifier parameters (conditional sampling with fixed WT_sample)
    names = m1.keys() #Name of all mutant parameters
    params = len(m1.columns)
    M1_mods_matrix = np.empty(shape=(params,1000), dtype=float)
    i = 0
    for name in names:
        M1_mods_matrix[i] = m1[name].to_numpy() #convert each modifier into a row in a matrix 
        i = i+1
    
    M1_mean_list = []
    j = 0

    for m in M1_mods_matrix: #calculate the mean of each modifier parameter
        means = sum(m)
        means = means/len(m)
        M1_mean_list.append(means)
        j = j+1

    #get wildtypes for m1
    WT1 = m1_wt.iloc[:,4:] #Calculate the mean for the m1 wildtypes for conditional distribution

    WT1_names = WT1.keys() 
    WT1_params = len(WT1.columns)
    WT1_matrix = np.empty(shape=(WT1_params,1000), dtype=float)
    i = 0
    for name in WT1_names:
        WT1_matrix[i] = WT1[name].to_numpy()
        i = i+1
    WT1_mean_list = []
    j = 0
    for m in WT1_matrix:
        means = sum(m)
        means = means/len(m)
        WT1_mean_list.append(means)
        j = j+1

    #Generate covariance matrix of full mutant params
    names = m1_wt.keys()
    params = len(m1_wt.columns)
    M1_matrix = np.empty(shape=(params,1000), dtype=float)
    i = 0
    for name in names:
        M1_matrix[i] = m1_wt[name].to_numpy()
        i = i+1
    M1_cov_matrix = np.cov(M1_matrix, bias = True)
    mu1 = M1_mean_list
    mu2 = WT1_mean_list
    C11 = M1_cov_matrix[0:4,0:4]
    C12 = M1_cov_matrix[0:4:,4:]
    C21 = M1_cov_matrix[4:,0:4]
    C22 = M1_cov_matrix[4:,4:]
    C22inv = np.linalg.inv(C22)
    a_minus_mu = (WT_sample - mu2)
    a_minus_mu[:, np.newaxis]
    C12C22inv = np.dot(C12,C22inv.T) 
    temp = np.dot(C12C22inv, a_minus_mu[:, np.newaxis])
    conditional_mu = [x+y for x, y in zip(mu1,temp.flatten().tolist())] 

    conditional_cov = C11 - np.dot(C12C22inv, C21)

    M1_multi_dis = multivariate_normal(mean = conditional_mu,
                                        cov = conditional_cov, 
                                        allow_singular = True
                                                 )
    
    M1_cond_params = M1_multi_dis.rvs(size = 100, random_state=rndint+ timeseed) #sample 100 m1 modifiers from conditional dist.
    M1s = pd.DataFrame(M1_cond_params, columns = m1.keys()) #convert to dataframe with column names as the m1 names
    ######################
    #mut2
    names = m2.keys()
    params = len(m2.columns)
    M2_mods_matrix = np.empty(shape=(params,1000), dtype=float)
    i = 0
    for name in names:
        M2_mods_matrix[i] = m2[name].to_numpy()
        i = i+1
    
    M2_mean_list = []
    j = 0

    for m in M2_mods_matrix:
        means = sum(m)
        means = means/len(m)
        M2_mean_list.append(means)
        j = j+1

    WT2 = m2_wt.iloc[:,4:]

    WT2_names = WT2.keys() 
    WT2_params = len(WT2.columns)
    WT2_matrix = np.empty(shape=(WT2_params,1000), dtype=float)
    i = 0
    for name in WT2_names:
        WT2_matrix[i] = WT2[name].to_numpy()
        i = i+1
    WT2_mean_list = []
    j = 0
    for m in WT2_matrix:
        means = sum(m)
        means = means/len(m)
        WT2_mean_list.append(means)
        j = j+1

    #Generate covariance matrix of full mutant params
    names = m2_wt.keys()
    params = len(m2_wt.columns)
    M2_matrix = np.empty(shape=(params,1000), dtype=float)
    i = 0
    for name in names:
        M2_matrix[i] = m2_wt[name].to_numpy()
        i = i+1
    M2_cov_matrix = np.cov(M2_matrix, bias = True)
    mu1 = M2_mean_list
    mu2 = WT2_mean_list
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
    M2s = pd.DataFrame(M2_cond_params, columns = m2.keys())
    # 

    mods_df = pd.DataFrame({"MAs":[], #dataframe for modifier parameters
                        "MBs":[],
                        "MCs":[],
                        "MNs":[],
                        "MAr":[],
                        "MBr":[],
                        "MCr":[],
                        "MNr":[],
                        "MAo":[],
                        "MBo":[],
                        "MCo":[],
                        "MNo":[]})
    
    WT_df = pd.DataFrame(WT_sample).transpose()
    WT_df.columns = ['As','Bs','Cs','Ns','Ar','Br','Cr','Nr','Ao','Bo','Co','Ck','No','Fo']

    mods_df[m1.keys()[0]] = M1s[m1.keys()[0]] #append the correct modifier columns to the mods dataframe
    mods_df[m1.keys()[1]] = M1s[m1.keys()[1]]
    mods_df[m1.keys()[2]] = M1s[m1.keys()[2]]
    mods_df[m1.keys()[3]] = M1s[m1.keys()[3]]
    mods_df[m2.keys()[0]] = M2s[m2.keys()[0]]
    mods_df[m2.keys()[1]] = M2s[m2.keys()[1]]
    mods_df[m2.keys()[2]] = M2s[m2.keys()[2]]
    mods_df[m2.keys()[3]] = M2s[m2.keys()[3]]

    

    mods_df = mods_df.replace(np.nan,0) #replace modifier params of wt node as 0.0 so it becomes 1 when un-logged

    WT_df = WT_df.append([WT_df]*(len(M1s[m1.keys()[0]])-1),ignore_index=True) #make wt_df the same length as mods_df

    log10_pars = pd.concat([WT_df,mods_df], axis=1) #returns full matrix with all WT and modifiers from this round of selection.
      
    return log10_pars

#%%

'''Save all the parameters for all pairwise mutants'''
prior_mutant = 'R7_O2'
size = 100
DM_names = DM_stripes['genotype'].tolist()
DM_names = list(set(DM_names[3:]))
DM_names.sort()

if prior_mutant == None: #incase code breaks, put in the genotype of last succesful run e.g R10_S10
        mutant_range:slice=slice(0,len(DM_names))
        count = 0
else:
    position = DM_names.index(prior_mutant)
    count = position
    mutant_range:slice=slice(position+1,len(DM_names))

for genotypes in DM_names[mutant_range]:
    bar = pb.ProgressBar(maxval = size).start() #import time
    #get genotypeID
    mutant_list = get_mut_names(genotypes)
    log_params = get_dm_params(mutant_list) #100 parameter sets, one wt sample.
    for i in range(0,size):
        temp = get_dm_params(mutant_list)
        bar.update(i+1)
        log_params = pd.concat([log_params,temp], ignore_index=True)

    path = f'../results/New_params/Pairwise_params/{genotypes}.csv'
    count = count + 1
    log_params.to_csv(path, index = False)
    print('mutant ', genotypes, 'completed, ', count, 'out of 300')

#%%
'''Visualising fits of pairwise predictions'''
def Visualise_mut(mutants:list):
    mut1 = mutants[0]
    mut2 = mutants[1]
    genotype = get_mut_ids(mutants)
    path = f'../results/New_params/Pairwise_params/{genotype}.csv'

    pars_df = pd.read_csv(path)
    pars_df = pars_df.head(10000)
    pars_df = 10**pars_df


    DM_df = meta_dict['DM']
    pair_mut_dict = DM_df[DM_df['genotype'] == genotype]

    hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)
    # np.random.seed(0)  

    set_list = []
    data = meta_dict['WT']
    pairwise_inducer = [0.00001, 0.0002, 0.2]
    ind = pd.DataFrame(pairwise_inducer)
    low = []
    med = []
    high = []

    for index, row in pars_df.iterrows():
        par_list = row.tolist()
        Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model_muts2(I_conc= ind,params_list=par_list)

        low.append(Stripe_est_array.iloc[0,0])
        med.append(Stripe_est_array.iloc[1,0])
        high.append(Stripe_est_array.iloc[2,0])

    data = {'low':np.log10(low), 'medium':np.log10(med), 'high':np.log10(high)}
    fluo_df = pd.DataFrame(data)
    fig, axes = plt.subplots(figsize=(10,6))

    axes2 = axes.twinx()
    point = []
    SD = []
    for obs_m, obs_sd in zip(pair_mut_dict['obs_fluo_mean'],pair_mut_dict['obs_SD']):
        point.append(obs_m)
        SD.append(obs_sd)

    point = np.log10(point)
    SD = np.log10(SD)
    sns.violinplot(data=fluo_df, ax=axes, orient='v', color = 'mistyrose' )
    sns.pointplot(x=np.arange(len(point)), y=point, ax=axes2, color = 'darkcyan')
    axes2.set_ylim(axes.get_ylim())

    data = meta_dict['WT']
    data_stripe = [data.Stripe[1],data.Stripe[5],data.Stripe[14],]
    data_stripe = np.log10(data_stripe)
    sns.pointplot(x=np.arange(len(data_stripe)), y=data_stripe, ax=axes2, color = 'indigo')

    Rand = mpatches.Patch(color= 'mistyrose', label='Estimated fluorescence')
    Wildtype = mpatches.Patch(color= 'indigo', label='Wildtype') #Could potenitally plot the actual wildtype data
    data_set = mpatches.Patch(color= 'darkcyan', label='Pairwise data')
    plt.legend(handles=[data_set,Rand,Wildtype], bbox_to_anchor=(1, 1), title = "Legend")
    plt.title(f'Pairwise mutant fit: {genotype}')
    axes.set_xlabel('Inducer Concetration')
    axes.set_ylim(2,5)
    axes2.set_ylim(2,5)
    axes.set_ylabel('Log_Fluorescence')
    axes.set_xticks(ticks=range(len(fluo_df.columns)), labels=fluo_df.columns)
    plt.show()
    return

#muts = ['Regulator10','Output2']
#Visualise_mut(mutants = muts)

#%%
'''Get the median epistasis per mutant and plot that'''

## Initizalization of dummy variables (parameters of the model) for sympy
import numpy as np 
import seaborn as sns
import sympy as sym
from os import listdir
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from os import listdir
from scipy import stats
symbolnames = ['aR','aS','aO','cS','cR','cO','bR','bS','bO','nR','nS','nO','I']
basepars = ['A','B','C','N'] # nbame of the pars for each mutant
myVars = vars()
for symbolname in symbolnames:
    myVars[symbolname] = sym.symbols(symbolname)
sym.init_printing(use_unicode=True)

def stst(pars):
    ''' 
    returns steady state values of the ODE given a parameter set
    for the three nodes S,R,O. It also returns Oh which is the
    expected output fluorescence in the absence of regulator
    '''


    affS = np.power(pars['Cs']*pars['I'],pars['Ns'])
    Seq = (pars['As'] + pars['Bs']*affS)/(1+affS)

    affR = np.power(pars['Cr']*Seq,pars['Nr'])
    Req = pars['Ar'] + pars['Br']/(1+affR)

    affO = np.power(pars['Co']*(Req+pars['Ck']*Seq),pars['No'])
    Oeq =  (pars['Ao'] + pars['Bo']/(1+affO))*pars['Fo'] #multiply by F_o

    affOh = np.power(pars['Co']*Req,pars['No']) #halfoutput is only sensor
    Oheq = pars['Ao'] + pars['Bo']/(1+affOh)

    return Seq,Req,Oeq,Oheq

mut_list = ['Sensor10','Output1']
# # modes = Compare_Epistasis('R10_O1.csv', mut_list)
file = 'S10_O1.csv'
Visualise_SM_fit(mut_name=mut_list[0],iter = 'final', plot_num= 50, save=False)
Visualise_SM_fit(mut_name=mut_list[1],iter = 'final', plot_num= 50, save=False)
Visualise_mut(mutants = mut_list)

#%%
# def Compare_Epistasis(file, mut_list):
#Try all parameters as using 10,000 only contains 20 WT sets. Should I try to do a conditional selection using a 1WT:1M ratio?
depth = 10000 # MAX = 250000, number of parameter sets to use per file
folder = '../results/New_params/Pairwise_params/'
epi_model = pd.DataFrame()
mode_epi = pd.DataFrame()

letters = r'[a-zA-Z]'
genotype = get_mut_ids(mut_list)
DM_df = meta_dict['DM']
pair_mut_dict = DM_df[DM_df['genotype'] == genotype]
mut1_data = get_data_SM(mut_list[0])
mut2_data = get_data_SM(mut_list[1])

# for file in listdir(folder):

# Extract from file title, which is the duplet
mutant_letters = re.findall(letters, file)
m1_str = mutant_letters[0].lower() # first character of filename string
m2_str = mutant_letters[1].lower() # first character of filename string
print('mutant_combo: {} with mutants {} and {}'.format(file,m1_str,m2_str))

# Calulate WT fluorescence
data_WT = pd.read_csv(folder+file)
data_WT = data_WT.head(depth) # if depth<10000 then only a subset of parameters is loaded
data_WT = 10**data_WT #convert from log10
data_WT['I'] = 0.0002 # WT peak position
WT_data = meta_dict['WT']
WT_fluo = stst(data_WT)[2]
plt.hist(np.log10(WT_fluo), bins='auto')
plt.axvline(x=np.log10(WT_data.iloc[5].Stripe), c = 'red', label = 'WT data') #plot stripe medium inducer conc
plt.xlabel('log10(fluorescence)')
plt.ylabel('density')
plt.title('m1_fluo')
plt.legend()
plt.show()

# Creating dataframes for singlets and duplet, in these dataframes the paremets will be modified using the fitted modifiers
data_mut1 = data_WT.copy()
data_duplet = data_WT.copy()
data_mut2 = data_WT.copy()

# Mutant 1 and duplet part
for par in basepars: # for each parameter of a node
    data_mut1[par+m1_str] = data_mut1[par+m1_str]*data_mut1['M'+par+m1_str]
    data_duplet[par+m1_str] = data_duplet[par+m1_str]*data_duplet['M'+par+m1_str]
# if m1_str == 'o': # in the case that 1 of the mutatns is the outout, apply fluorescence correction. Note that this should not affect epistasis
#     data_mut1['Fo'] = data_mut1['F_o']
#     data_duplet['Fo'] = data_duplet['F_o']

m1_fluo = stst(data_mut1)[2]
plt.hist(np.log10(m1_fluo), bins='auto')
plt.axvline(x=np.log10(mut1_data.iloc[5].Stripe), c = 'red', label = 'SM data') #plot stripe medium inducer conc
plt.xlabel('log10(fluorescence)')
plt.ylabel('density')
plt.title('m1_fluo')
plt.legend()
plt.show()


# Mutant 2 and duplet part
for par in basepars:
    data_mut2[par+m2_str] = data_mut2[par+m2_str]*data_mut2['M'+par+m2_str]
    data_duplet[par+m2_str] = data_duplet[par+m2_str]*data_duplet['M'+par+m2_str]
# if m2_str == 'o':
#     data_mut2['F_o'] = data_mut2['F_o']*data_mut2['MF_o']
#     data_duplet['F_o'] = data_duplet['F_o']*data_duplet['MF_o']

exp_duplet_fluo = pair_mut_dict['obs_fluo_mean']
m2_fluo = stst(data_mut2)[2] 
plt.hist(np.log10(m2_fluo), bins='auto')
plt.axvline(x=np.log10(mut1_data.iloc[5].Stripe), c = 'red',  label = 'SM data')
plt.title('m2_fluo')
plt.xlabel('log10(fluorescence)')
plt.ylabel('density')
plt.legend()
plt.show()
# Duplet
duplet_fluo = stst(data_duplet)[2]
#Method1: take the median predicted value and calculate the epistasis from there.
#   Maybe take only the 5 percent of fluorescent values around the mean and calculate epistasis as those are the more 'accurate' simulations.
plt.hist(np.log10(duplet_fluo), bins='auto')
plt.title('duplex_fluo')
plt.xlabel('log10(fluorescence)')
plt.ylabel('density')
plt.axvline(x = np.log10(exp_duplet_fluo.iloc[1]),c = 'r', label = 'Pairwise data')
plt.legend()
plt.show()



#Method2: take the median of the epistasis values and plot them
#mean of -0.004 where we want a mean of -0.1 ish...

logG_expected = np.log10(m1_fluo/WT_fluo) + np.log10(m2_fluo/WT_fluo)
logG_model =  np.log10(duplet_fluo/WT_fluo)
Epistasi = logG_model - logG_expected

neg_fluo = []
for index,epi in enumerate(Epistasi):
    if epi < 0:
        neg_fluo.append(duplet_fluo[index])
kde = gaussian_kde(neg_fluo)
x = np.linspace(min(neg_fluo),max(neg_fluo),num=1000)
y = kde(x)
mode_index = np.argmax(y)
neg_mode = x[mode_index]
plt.hist(np.log10(neg_fluo))
plt.title('negative eps duplex_fluo')
plt.xlabel('log10(fluorescence)')
plt.ylabel('density')
plt.axvline(x = np.log10(exp_duplet_fluo.iloc[1]),c = 'r', label = 'Pairwise data')
plt.axvline(x = np.log10(neg_mode),c = 'black', label = 'Mode')
plt.legend()
plt.show()



pos_fluo = []
for index,epi in enumerate(Epistasi):
    if epi > 0:
        pos_fluo.append(duplet_fluo[index])
kde = gaussian_kde(pos_fluo)
x = np.linspace(min(pos_fluo),max(pos_fluo),num=1000)
y = kde(x)
mode_index = np.argmax(y)
pos_mode = x[mode_index]
plt.hist(np.log10(pos_fluo))
plt.title('positive eps duplex_fluo')
plt.xlabel('log10(fluorescence)')
plt.ylabel('density')
exp_duplet_fluo = pair_mut_dict['obs_fluo_mean']
plt.axvline(x = np.log10(exp_duplet_fluo.iloc[1]),c = 'r', label = 'Pairwise data')
plt.axvline(x = np.log10(pos_mode),c = 'black', label = 'Mode')
plt.legend()
plt.show()

kde = gaussian_kde(Epistasi)
x = np.linspace(min(Epistasi),max(Epistasi),num=1000)
y = kde(x)
mode_index = np.argmax(y)
mode = pd.DataFrame({'Mode_Eps':[]})
temp = pd.DataFrame({'Mode_Eps':[x[mode_index]]})
mode = pd.concat([mode,temp], ignore_index=True)

mode_epi = pd.concat([mode_epi, mode])
# epi_model = pd.concat([epi_model,Epistasis])

# plt.xlim([-0.5,0.5])    
# plt.hist(epi_model,bins = 100, range=[-0.5,0.5])
# plt.show()
plt.xlim([-0.5,0.5])    
plt.hist(Epistasi,bins = 'auto', range=[-0.5,0.5])
plt.axvline(x = mode_epi.iloc[0][0], c = 'r', label = 'mode')
plt.title('Epistasis')
plt.xlabel('Epistasis')
plt.ylabel('density')
plt.legend()
plt.show()
# return mode_epi
    


    
# %%
def Compare_Epistasis():
    depth = 10000 # MAX = 250000, number of parameter sets to use per file
    folder = '../results/New_params/Pairwise_params/'
    epi_model = pd.DataFrame()
    mode_epi = pd.DataFrame()

    letters = r'[a-zA-Z]'
    # genotype = get_mut_ids(mut_list)
    DM_df = meta_dict['DM']
    #pair_mut_dict = DM_df[DM_df['genotype'] == genotype]
    # mut1_data = get_data_SM(mut_list[0])
    # mut2_data = get_data_SM(mut_list[1])

    for file in listdir(folder):

        # Extract from file title, which is the duplet
        mutant_letters = re.findall(letters, file)
        m1_str = mutant_letters[0].lower() # first character of filename string
        m2_str = mutant_letters[1].lower() # first character of filename string
        print('mutant_combo: {} with mutants {} and {}'.format(file,m1_str,m2_str))

        # Calulate WT fluorescence
        data_WT = pd.read_csv(folder+file)
        data_WT = data_WT.head(depth) # if depth<10000 then only a subset of parameters is loaded
        data_WT = 10**data_WT #convert from log10
        data_WT['I'] = 0.0002 # WT peak position
        WT_fluo = stst(data_WT)[2]


        # Creating dataframes for singlets and duplet, in these dataframes the paremets will be modified using the fitted modifiers
        data_mut1 = data_WT.copy()
        data_duplet = data_WT.copy()
        data_mut2 = data_WT.copy()

        # Mutant 1 and duplet part
        for par in basepars: # for each parameter of a node
            data_mut1[par+m1_str] = data_mut1[par+m1_str]*data_mut1['M'+par+m1_str]
            data_duplet[par+m1_str] = data_duplet[par+m1_str]*data_duplet['M'+par+m1_str]
        # if m1_str == 'o': # in the case that 1 of the mutatns is the outout, apply fluorescence correction. Note that this should not affect epistasis
        #     data_mut1['Fo'] = data_mut1['F_o']
        #     data_duplet['Fo'] = data_duplet['F_o']

        m1_fluo = stst(data_mut1)[2]
        # plt.hist(np.log10(m1_fluo), bins='auto')
        # plt.axvline(x=np.log10(mut1_data.iloc[5].Stripe), c = 'red') #plot stripe medium inducer conc
        # plt.title('m1_fluo')
        # plt.show()


        # Mutant 2 and duplet part
        for par in basepars:
            data_mut2[par+m2_str] = data_mut2[par+m2_str]*data_mut2['M'+par+m2_str]
            data_duplet[par+m2_str] = data_duplet[par+m2_str]*data_duplet['M'+par+m2_str]
        # if m2_str == 'o':
        #     data_mut2['F_o'] = data_mut2['F_o']*data_mut2['MF_o']
        #     data_duplet['F_o'] = data_duplet['F_o']*data_duplet['MF_o']

        # exp_duplet_fluo = pair_mut_dict['obs_fluo_mean']
        m2_fluo = stst(data_mut2)[2] 
        # plt.hist(np.log10(m2_fluo), bins='auto')
        # plt.axvline(x=np.log10(mut1_data.iloc[5].Stripe), c = 'red')
        # plt.title('m2_fluo')
        # plt.show()

        # Duplet
        duplet_fluo = stst(data_duplet)[2]
        # #Method1: take the median predicted value and calculate the epistasis from there.
        # #   Maybe take only the 5 percent of fluorescent values around the mean and calculate epistasis as those are the more 'accurate' simulations.
        # plt.hist(np.log10(duplet_fluo), bins='auto')
        # plt.title('duplex_fluo')
        # plt.axvline(x = np.log10(exp_duplet_fluo.iloc[1]),c = 'r')
        # plt.show()



        #Method2: take the median of the epistasis values and plot them
        #mean of -0.004 where we want a mean of -0.1 ish...

        logG_expected = np.log10(m1_fluo/WT_fluo) + np.log10(m2_fluo/WT_fluo)
        logG_model =  np.log10(duplet_fluo/WT_fluo)
        Epistasi = logG_model - logG_expected

        # neg_fluo = []
        # for index,epi in enumerate(Epistasi):
        #     if epi < 0:
        #         neg_fluo.append(duplet_fluo[index])
        # kde = gaussian_kde(neg_fluo)
        # x = np.linspace(min(neg_fluo),max(neg_fluo),num=1000)
        # y = kde(x)
        # mode_index = np.argmax(y)
        # neg_mode = x[mode_index]
        # plt.hist(np.log10(neg_fluo))
        # plt.title('negative eps duplex_fluo')
        # plt.axvline(x = np.log10(exp_duplet_fluo.iloc[1]),c = 'r')
        # plt.axvline(x = np.log10(neg_mode),c = 'black')
        # plt.show()



        # pos_fluo = []
        # for index,epi in enumerate(Epistasi):
        #     if epi > 0:
        #         pos_fluo.append(duplet_fluo[index])
        # kde = gaussian_kde(pos_fluo)
        # x = np.linspace(min(pos_fluo),max(pos_fluo),num=1000)
        # y = kde(x)
        # mode_index = np.argmax(y)
        # pos_mode = x[mode_index]
        # plt.hist(np.log10(pos_fluo))
        # plt.title('positive eps duplex_fluo')
        # exp_duplet_fluo = pair_mut_dict['obs_fluo_mean']
        # plt.axvline(x = np.log10(exp_duplet_fluo.iloc[1]),c = 'r')
        # plt.axvline(x = np.log10(pos_mode),c = 'black')
        # plt.show()

        kde = gaussian_kde(Epistasi)
        x = np.linspace(min(Epistasi),max(Epistasi),num=1000)
        y = kde(x)
        mode_index = np.argmax(y)
        mode = pd.DataFrame({'Mode_Eps':[]})
        temp = pd.DataFrame({'Mode_Eps':[x[mode_index]]})
        mode = pd.concat([mode,temp], ignore_index=True)

        mode_epi = pd.concat([mode_epi, mode])
        epi_model = pd.concat([epi_model,Epistasi])

        # plt.xlim([-0.5,0.5])    
        # plt.hist(epi_model,bins = 100, range=[-0.5,0.5])
        # plt.show()
        # plt.xlim([-0.5,0.5])    
        # plt.hist(Epistasi,bins = 'auto', range=[-0.5,0.5])
        # plt.axvline(x = mode_epi.iloc[0][0], c = 'r')
        # plt.show()
    
    plt.hist(epi_model, bins = 'auto', density = True)
    plt.title('Epistasis of all pairwise mutants')
    plt.xlabel('Epistasis')
    plt.ylabel('Density')
    plt.show()
    plt.hist(mode_epi, bins = 'auto', density = True)
    plt.title('Mode Epistasis of all pairwise mutants')
    plt.xlabel('Mode Epistasis')
    plt.ylabel('Density')
    plt.show()

    return mode_epi, epi_model

mode_epi,epi_model = Compare_Epistasis()


# %%
 names2 = m2.keys()
    # params2 = len(m2.columns)
    # M2_mods_matrix = np.empty(shape=(params2,1000), dtype=float)
    # i = 0
    # for name2 in names2:
    #     M2_mods_matrix[i] = m2[name2].to_numpy()
    #     i = i+1
    
    # M2_mean_list = []
    # j = 0

    # for m_2 in M2_mods_matrix:
    #     means2 = sum(m_2)
    #     means2 = means2/len(m_2)
    #     M2_mean_list.append(means2)
    #     j = j+1

    # #get wildtypes for m1
    # WT2 = m2_wt.iloc[:,4:]

    # WT2_names = WT2.keys() 
    # WT2_params = len(WT2.columns)
    # WT2_matrix = np.empty(shape=(WT2_params,1000), dtype=float)
    # i = 0
    # for name2 in WT2_names:
    #     WT2_matrix[i] = WT2[name2].to_numpy()
    #     i = i+1
    # WT2_mean_list = []
    # j = 0
    # for m_2 in WT2_matrix:
    #     means2 = sum(m_2)
    #     means2 = means2/len(m_2)
    #     WT2_mean_list.append(means2)
    #     j = j+1

    # #Generate covariance matrix of full mutant params
    # names2 = m2_wt.keys()
    # params2 = len(m2_wt.columns)
    # M2_matrix = np.empty(shape=(params2,1000), dtype=float)
    # i = 0
    # for name2 in names2:
    #     M2_matrix[i] = m2_wt[name2].to_numpy()
    #     i = i+1
    # M2_cov_matrix = np.cov(M2_matrix, bias = True)
    # mu1 = M2_mean_list
    # mu2 = WT2_mean_list
    # C11 = M2_cov_matrix[0:4,0:4]
    # C12 = M2_cov_matrix[0:4:,4:]
    # C21 = M2_cov_matrix[4:,0:4]
    # C22 = M2_cov_matrix[4:,4:]
    # C22inv = np.linalg.inv(C22)
    # a_minus_mu = (WT_sample - mu2)
    # a_minus_mu[:, np.newaxis]
    # C12C22inv = np.dot(C12,C22inv.T) #not sure if transpose is correct
    # temp = np.dot(C12C22inv, a_minus_mu[:, np.newaxis])
    # conditional_mu = [x+y for x, y in zip(mu1,temp.flatten().tolist())]

    # conditional_cov = C11 - np.dot(C12C22inv, C21)

    # M2_multi_dis = multivariate_normal(mean = conditional_mu,
    #                                     cov = conditional_cov, 
    #                                     allow_singular = True
    #                                              )
    
    # M2_cond_params = M2_multi_dis.rvs(size = 100, random_state=rndint+ timeseed)

    # names = m2.keys()
    # params = len(m2.columns)
    # M2_mods_matrix = np.empty(shape=(params,1000), dtype=float)
    # i = 0
    # for name in names:
    #     M2_mods_matrix[i] = m2[name].to_numpy()
    #     i = i+1
    
    # M2_mean_list = []
    # j = 0

    # for m in M2_mods_matrix:
    #     means = sum(m)
    #     means = means/len(m)
    #     M2_mean_list.append(means)
    #     j = j+1

    # #get wildtypes for m1
    # WT2 = m2_wt.iloc[:,4:]

    # WT2_names = WT2.keys() 
    # WT2_params = len(WT2.columns)
    # WT2_matrix = np.empty(shape=(WT2_params,1000), dtype=float)
    # i = 0
    # for name in WT2_names:
    #     WT2_matrix[i] = WT2[name].to_numpy()
    #     i = i+1
    # WT2_mean_list = []
    # j = 0
    # for m in WT2_matrix:
    #     means = sum(m)
    #     means = means/len(m)
    #     WT2_mean_list.append(means)
    #     j = j+1

    # #Generate covariance matrix of full mutant params
    # names = m2_wt.keys()
    # params = len(m2_wt.columns)
    # M2_matrix = np.empty(shape=(params,1000), dtype=float)
    # i = 0
    # for name in names:
    #     M2_matrix[i] = m2_wt[name].to_numpy()
    #     i = i+1
    # M2_cov_matrix = np.cov(M2_matrix, bias = True)
    # mu1 = M2_mean_list
    # mu2 = WT2_mean_list
    # C11 = M2_cov_matrix[0:4,0:4]
    # C12 = M2_cov_matrix[0:4:,4:]
    # C21 = M2_cov_matrix[4:,0:4]
    # C22 = M2_cov_matrix[4:,4:]
    # C22inv = np.linalg.inv(C22)
    # a_minus_mu = (WT_sample - mu2)
    # a_minus_mu[:, np.newaxis]
    # C12C22inv = np.dot(C12,C22inv.T) #not sure if transpose is correct
    # temp = np.dot(C12C22inv, a_minus_mu[:, np.newaxis])
    # conditional_mu = [x+y for x, y in zip(mu1,temp.flatten().tolist())]

    # conditional_cov = C11 - np.dot(C12C22inv, C21)

    # M2_multi_dis = multivariate_normal(mean = conditional_mu,
    #                                     cov = conditional_cov, 
    #                                     allow_singular = True
    #                                              )
    
    # M2_cond_params = M2_multi_dis.rvs(size = 100, random_state=rndint+ timeseed)

    ######
    # mod_names2 = m2.keys()
    # mod_params2 = len(m2.columns)
    # M2_mods_matrix = np.empty(shape=(mod_params2,1000), dtype=float)
    # i = 0
    # for name in mod_names2:
    #     M2_mods_matrix[i] = m2[name].to_numpy()
    #     i = i+1
    # M2_mean_list = []
    # j = 0

    # for m in M2_mods_matrix:
    #     means = sum(m)
    #     means = means/len(m)
    #     M2_mean_list.append(means)
    #     j = j+1

    
    # #get wildtypes for m2
    # WT2 = m2_wt.iloc[:,4:]

    # WT2_names = WT2.keys() 
    # WT2_params = len(WT2.columns)
    # WT2_matrix = np.empty(shape=(WT2_params,1000), dtype=float)
    # i = 0
    # for name in WT2_names:
    #     WT2_matrix[i] = WT2[name].to_numpy()
    #     i = i+1
    # WT2_mean_list = []
    # j = 0
    # for m in WT2_matrix:
    #     means = sum(m)
    #     means = means/len(m)
    #     WT2_mean_list.append(means)
    #     j = j+1


    # #Generate covariance matrix of full mutant params
    # mut_names = m2_wt.keys()
    # mut_params = len(m2_wt.columns)
    # M2_matrix = np.empty(shape=(mut_params,1000), dtype=float)
    # i = 0
    # for name in mut_names:
    #     M2_matrix[i] = m2_wt[name].to_numpy()
    #     i = i+1
    # # m2_wt = m2_wt.T
    # M2_matrix = M2_matrix
    # # M2_cov_matrix = np.cov(m2_wt.values) #, bias = True)
    # M2_cov_matrix = np.cov(M2_matrix) #, bias = True)
    # mu1 = np.array(M2_mean_list)
    # mu2 = np.array(WT2_mean_list) #WT1_mean_list
    # C11 = M2_cov_matrix[0:4,0:4]
    # C12 = M2_cov_matrix[0:4:,4:]
    # C21 = M2_cov_matrix[4:,0:4]
    # C22 = M2_cov_matrix[4:,4:]
    # C22inv = np.linalg.inv(C22)
    # a_minus_mu = (WT_sample - mu2)
    # a_minus_mu[:, np.newaxis]
    # C12C22inv = np.dot(C12,C22inv) #original 
    # temp = np.dot(C12C22inv, a_minus_mu[:, np.newaxis])
    # #temp = np.dot(C12C22inv, a_minus_mu[:, np.newaxis]) #original
    # conditional_mu = [x+y for x, y in zip(mu1,temp.flatten().tolist())]
    # conditional_cov = C11 - np.dot(C12C22inv, C21)

    # M2_multi_dis = multivariate_normal(mean = conditional_mu,
    #                                     cov = conditional_cov,
    #                                     allow_singular = True)
    
    # M2_cond_params = M2_multi_dis.rvs(size = 500, random_state=rndint+ timeseed)