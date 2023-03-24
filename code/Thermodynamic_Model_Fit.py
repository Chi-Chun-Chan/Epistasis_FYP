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
#%%
therm_model=model_thermodynamic(params_list=[1]*14,I_conc=meta_dict["WT"].S)
func=therm_model.model
params_therm_dict={"sen_params":{"P_b":6.70063321e-02,"P_u":9.26794143e-07,"K_12":2.23007522e+03,"C_pa":1.84833176e+09,"A_s":1.45247730e+04},"reg_params":{"P_r":1.67754325e+00,"C_pt":1.26659501e-02,"K_t":6.16701374e-05,"A_r":1.09634721e+05},"out_h_params":{},"out_params":{"P_o":1.67754325e+00,"C_pl":0.00575464637, "K_l":2.40059869e-06,"A_o":7.98011665e+04},"free_params":{},"fixed_params":{"F_o":1.48194306e+00}}
params_therm_list=dict_to_list(params_therm_dict)
params_therm_list=get_WT_params(model_type=func,start_guess=params_therm_list,n_iter=1e5,method="Nelder-Mead",params_dict=params_therm_dict,custom_settings=[[1,0,0,0,0],[None,1,1,None,None],["C_pa","C_pt","C_pl","P_p","F_o"]],tol=1)
#%%
#good wild type fit, now fitting to single mutants
converged_params_therm_list=params_therm_list
converged_params_therm_dict=list_to_dict(old_dict=params_therm_dict,new_values=converged_params_therm_list)

model_fitting_SM(model_type=func,n_iter=1e5,params_dict=converged_params_therm_dict)