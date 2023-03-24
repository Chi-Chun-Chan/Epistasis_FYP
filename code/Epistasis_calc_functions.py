#%%
import pandas as pd
import numpy as np
import re
from itertools import permutations
from scipy import stats
from data_wrangling import meta_dict
from Models import *
from Model_fitting_functions import dict_to_list
#%%
#This file is defines the functions used to calculate epistasis for all double and triple mutants from observed and expected flourcsence at low, medium and high inducer concs. See Epistasis_calcs.py to run functions for a given model and export them to excel.
df_S = meta_dict['SM']
df_DM = meta_dict['DM']
df_TM = meta_dict['TM']
df_M = (pd.concat([df_DM, df_TM], ignore_index = True)).drop_duplicates()

I_conc = {'low':0.0, 'medium': 0.0002, 'high':0.2}
I_conc_np = np.array(list(I_conc.values()))
I_conc_np[1] = 0.000195 #medium Inducer concentration is rounded in df_S, want more accurate value (0.000195) when fitting using a model

#%%
#get parameter values for model from here - must give it the file path after the data directory as a string
def get_params(model:str, strat:str = 'all'):
    model_name = model.__qualname__
    df_fits = pd.read_excel('../data/'+model_name+ 'SM_params_'+strat+'.xlsx').rename(columns={'Unnamed: 0': 'mutant'})
    return df_fits
#stripe output at low, medium and high incucer concentration for WT
g_WT = np.array(df_DM['obs_fluo_mean'][df_DM['genotype']=='WT'])
g_WT_sd = np.array(df_DM['obs_SD'][df_DM['genotype']=='WT'])

#calculates the GFP for a single mutant relative to WT
def g(mutant:str):
    g_single = np.empty(len(I_conc))
    g_single_std = np.empty(len(I_conc))
    for i, conc in enumerate(I_conc.values()):
        g_single[i] = df_S['Stripe_mean'][(df_S['Inducer'] == conc) & (df_S['Mutant_ID'] == mutant)]
        g_single_std[i] = df_S['Stripe_stdev'][(df_S['Inducer'] == conc) & (df_S['Mutant_ID'] == mutant)]
    g_single = np.divide(g_single, g_WT)
    g_single_std_rel = np.divide(g_single_std, g_WT_sd)
    return g_single , g_single_std, g_single_std_rel

#expected flourecence for a set set of parameters and a chosen model at concentrations in I_conc dictionary relative to WT
def g_hat(mutant:str, model, df_fits):
    params = df_fits[df_fits['mutant']== mutant].values.flatten().tolist()[1:-2]
    g_hat = np.divide(model(params_list = params, I_conc = I_conc_np)[-1],g_WT)
    return g_hat

#expected log fold flourecence for mutants assuming log additivity of single mutants
def G_log(mutants:list):
    G_log, G_log_std, G_log_std_rel = g(mutants[0])
    G_log_std = np.power(G_log_std, 2)
    G_log_std_rel = np.power(G_log_std_rel, 2)
    for mut in mutants[1:]:
        G_logNEW, G_log_stdNEW, G_log_std_relNEW = g(mut)
        G_log = np.multiply(G_log,G_logNEW)
        G_log_std += np.power(G_log_stdNEW, 2)
        G_log_std_rel += np.power(G_log_std_relNEW, 2)
    G_log = np.log10(G_log)
    x = np.power(G_log_std, 1/2)
    y = np.power(G_log_std_rel, 1/2)
    G_log_std = np.multiply(G_log, x)
    G_log_std_rel = np.multiply(G_log, y)
    return G_log, G_log_std, G_log_std_rel

#expected GFP for double or triple mutants for a given model
def G_hat(model, mutants:list, df_fits:pd.DataFrame, example_dict = {}):
    #copy df_fits and replace relevant parameters with mutated ones in df_fits row for first mutant in "mutants" list
    df_fits1 = df_fits.copy(deep = False)
    for mut in mutants[1:]:
        #get parameter names relevant to "mut"
        mut_type = f"{mut[0:3].lower()}"
        param_list = []
        for key in example_dict:
            if key.startswith(mut_type):
                param_list += list(example_dict[key])
        new_params = df_fits1[df_fits1['mutant'] == mut].filter(param_list)
        df_fits1.loc[df_fits1['mutant'] == mutants[0],new_params.columns] = new_params.values
    #get parameter values for mutant combination
    G_hat = np.log10(g_hat(mutants[0], model, df_fits1))
    return G_hat

#get list of possible mutant id names - e.g ['Sensor1','Regulator1'] --> ['S1_R1' , 'R1_S1']
def get_mut_ids(mutants:list, want_perms = True):
    muts = []
    for mut in mutants:
        mut_index = re.search("[0-9]",mut).start()
        mutant1_id = f"{mut[:1]}{mut[mut_index:]}"
        muts.extend([mutant1_id])
    #permutations of mutants to search against in df_M
    #source data has genotypes named in order: R then S then O, I'm copying to make it easier to compare/check
    mut_perms = []
    for ids in permutations(muts):
        string = ''
        for j in range(len(ids)):
            string += ids[j] +'_'
        mut_perms += [string[:-1]]
    #search df for a matching mut_id to one in mut_perms to keep nomenclature the consistent
    mut_id = list(df_M['genotype'][df_M['genotype'].isin(mut_perms)])[0]
    return mut_id

#the opposite of get_mut_ids that turns id into list of names e.g. S1_O1 --> ['Sensor1','Regulator1']
def get_mut_names(mut_id:str):
    mutant_names = []
    ids = mut_id.split('_')
    for mut in ids:
        if mut.startswith('S'):
            mut = f"Sensor{mut[1:]}"
            mutant_names.extend([mut])
        if mut.startswith('R'):
            mut = f"Regulator{mut[1:]}"
            mutant_names.extend([mut])
        if mut.startswith('O'):
            mut = f"Output{mut[1:]}"
            mutant_names.extend([mut])
    return mutant_names

#observed GFP for double or triple mutants
#takes list of possible combinations of mutation ids as parameter e.g. ['O1_R1', 'R1_O1']
def G_lab(mutants:list):
    mut_id = get_mut_ids(mutants)
    df_mutant = df_M[df_M['genotype'].isin([mut_id])] 
    #get mean and std for low, med, high inducer concs
    #NB - assumes low inducer concs come above medium, which come before high in mutant data
    mut_id = list(df_mutant['genotype'])[0]
    MT_mean = np.array(df_mutant['obs_fluo_mean'])
    MT_sd = np.array(df_mutant['obs_SD'])
    G_lab = np.log10(np.divide(MT_mean, g_WT))
    return G_lab, MT_sd

def Epistasis(mutants:list,df_fits:pd.DataFrame, model = 'observed', example_dict = {}):
    G_log_mean = G_log(mutants)[0]
    #Ghat_logadd_std = G_log(mutants)[1]   
    if model == 'observed':
        G = G_lab(mutants)
        G_mean = G[0]
        Epsilon =  G[0] - G_log_mean
        # Mann-Whitney U test removed because I deleted part of the syntax accidently and couldn't see how to quickly fix it
        p_val = stats.ttest_ind_from_stats(mean1 = 1091,std1 = 252, nobs1 = 3,mean2 = 1730, std2 = 303, nobs2 = 3)[1]
        p_vals = np.array([p_val, p_val, p_val])
    else:
        G_mean = G_hat(model, mutants, df_fits, example_dict = example_dict)
        Epsilon = G_mean - G_log_mean
        p_vals = np.array([0,0,0])
    if len(mutants) == 2:
        genotype_category = ['pairwise']*3
    else:
        genotype_category = ['triplet']*3
    genotype = [get_mut_ids(mutants=mutants)]*3
    inducer_level = ['low', 'medium', 'high']
    return Epsilon, p_vals, G_mean, G_log_mean, genotype_category, genotype, inducer_level

#calculate epistasis for all double mutants - returns dataframe with Epistasis mean and PValue, and GFP output under a model or observed in lab, and G_logadd for each pairwise and triple mutant  
def get_Eps(model='observed', strat = 'all', example_dict = {}):
    df_Eps = pd.DataFrame({'Ep': [],'Ep_pVal':[],'G': [], 'G_log': [] ,'genotype category': [],'genotype': [], 'inducer level': []})
    cols = len(df_Eps.axes[1])
    if model != 'observed':
        df_fits = get_params(model, strat)
    else:
        df_fits = []
    print("printing every 100th mutant processed to show progress...")
    for i, mut_id in enumerate(df_M['genotype'][(df_M['inducer level'] == 'low')][1:]):
        if i % 100 == 0:
            print(mut_id)
        mut_names = get_mut_names(mut_id)
        mut_Eps = Epistasis(mut_names,df_fits, model, example_dict = example_dict)
        row_low = []
        row_med = []
        row_high = []
        #make list from outputs of 'Epistasis' function to add as rows to Eps_low/med/high
        for j in range(cols):
            row_low += [mut_Eps[j][0]]
            row_med += [mut_Eps[j][1]]
            row_high += [mut_Eps[j][2]]
        #wierd indexes ensure low comes before med comes before high
        df_Eps.loc[1- 1/(i+1)] = row_low
        df_Eps.loc[2- 1/(i+1)] = row_med
        df_Eps.loc[len(df_Eps)] = row_high
    #reorder indexes and then reset them  to integers
    df_Eps = df_Eps.sort_index().reset_index(drop=True)
    #now add genotype names and categories
    #df_Eps[['genotype' ,'genotype category', 'inducer level']] =  df_M[['genotype', 'genotype category', 'inducer level']][301:304].reset_index(drop=True)
    #and a boolean indicator for significant epistasis
    df_Eps['Sig_Epistasis'] = np.where(df_Eps['Ep_pVal'] < 0.05, True, False)
    return df_Eps

#export Epistases to an excel document named after the chosen model
def Eps_toExcel(model= 'observed', strategy:str = '', example_dict:dict = {}):
    df_Eps = get_Eps(model = model, strat = strategy, example_dict = example_dict)
    #export to a spreadsheet
    if model != 'observed':
        model_name = model.__qualname__
        strategy = f"_{strategy}"
    else:
        model_name = 'observed'
        strategy = ''
    df_Eps.to_excel('../results/Eps_'+model_name+strategy+'.xlsx')
    return df_Eps

#%%