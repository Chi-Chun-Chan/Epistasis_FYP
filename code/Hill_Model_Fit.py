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
# dataframe = pd.read_csv('../data/smc/pars_final.out')
# dataf = pd.read_csv(dataframe, delimiter=' ', index_col=None, header=None)

path = '../data/smc_hill/pars_final.out'

df = Out_to_DF_hill(path, model_hill)

# par_list = list(df.columns)
# df = pd.melt(df, value_vars=par_list, var_name='params', value_name='Param value',ignore_index=False)


Paired_Density_plot(df)

#get (5?) lowest distances and index number then select that from pars_out as list



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
RSS_Score(params_hill_list,model_hill,data)
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
RSS_Score(params_list,model_hill,d)



# %%
