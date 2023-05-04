#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from Models import *
from Models import model_hill_shaky
from data_wrangling import *
from itertools import chain
from itertools import repeat
from PyPDF2 import PdfMerger
import inspect
from Model_fitting_functions import *
#%%
#%%

params_compdeg_dict={"sen_params":{"A_s":6.30849013e+02,"B_s":1.90729055e+04,"C_s":1.10540400e+03,"N_s":1.09318053e+00},"reg_params":{"B_r":1.31151789e+05,"C_r":2.61988491e-02,"N_r":1.09664020e+00},"out_h_params":{},"out_params":{"B_o":8.26418974e+03,"C_o":1.59326368e-04,"N_o":5.64975947e+00},"free_params":{ "F_o":1.07960469e+00, "K":1.2}}

temp = []
temp = [1.14630707e+03, 2.88086133e+04, 2.19631449e+03, 1.38878016e+00,
       6.26230093e+06, 4.41941132e+01, 5.06015544e-01, 3.00206662e+05,
       1.90307528e+02, 1.01135875e-01, 1.28218189e-01, 8.89160486e+15]
params_compdeg_dict = list_to_dict(params_compdeg_dict,temp)

params_compdeg_list=dict_to_list(params_compdeg_dict)

compdeg_model= CompDeg(params_list=params_compdeg_list, I_conc=meta_dict["WT"].S)
func = compdeg_model.model

converged_params_list_compdeg=Plotter(model_type=func,start_guess=params_compdeg_list,n_iter=1e5,method="Nelder-Mead",params_dict=params_compdeg_dict,custom_settings=[],tol=0.0001)

#%%
converged_params_list_compdeg=get_WT_params(model_type=func,start_guess=params_compdeg_list,n_iter=1e5,method="Nelder-Mead",params_dict=params_compdeg_dict,custom_settings=[],tol=0.0001)



# %%
converged_params_list_compdeg=params_compdeg_list
converged_params_compdeg_dict=list_to_dict(old_dict=params_compdeg_dict,new_values=converged_params_list_compdeg)
model_fitting_SM(model_type=func,n_iter=1e5,params_dict=converged_params_compdeg_dict)

#%%
