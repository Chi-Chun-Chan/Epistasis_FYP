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

# params_shaky_dict= {"sen_params":{"A_s":6.59203090e+02,"B_s":2.52693912e+04,"C_s":1.29142360e+03,"N_s":1.26521014e+00},"reg_params":{"A_r":1.81777999e+03,"B_r":4.20147672e+01,"C_r":3.84425469e-01,"N_r":4.43009968e-01},"out_h_params":{},"out_params":{"A_o":2.99771927e+03,"B_o":9.42809978e+03,"C_o":2.76353967e-01,"N_o":7.40963897e-01},"free_params":{"F_o":1.43207536e-02}}
params_shaky_dict={"sen_params":{"A_s":7.53988109e+02,"B_s":2.55645597e+04,"C_s":1.44279196e+03,"N_s":1.26717614e+00},"reg_params":{"A_r":1.70454779e+00,"B_r":1.31151789e+05,"C_r":2.61988491e-02,"N_r":1.09664020e+00},"out_h_params":{},"out_params":{"A_o":2.02624106e+02,"B_o":8.26418974e+03,"C_o":1.59326368e-04,"N_o":5.64975947e+00},"free_params":{"F_o":1.07960469e+00}}

temp = []
temp = [6.55427367e+02, 2.49568739e+04, 1.40128193e+03, 1.21549733e+00,
2.10196139e+00, 3.39690073e+04, 8.76696737e-02, 5.45810244e-01,
4.24382667e-01, 8.57979689e+03, 1.98355504e-04, 6.28186570e+00,
8.49908985e-01]
params_shaky_dict = list_to_dict(params_shaky_dict,temp)

params_shaky_list=dict_to_list(params_shaky_dict)

shaky_model= model_hill_shaky(params_list=params_shaky_list, I_conc=meta_dict["WT"].S)
func = shaky_model.model

converged_params_list_shaky=Plotter(model_type=func,start_guess=params_shaky_list,n_iter=1e5,method="Nelder-Mead",params_dict=params_shaky_dict,custom_settings=[],tol=0.0001)

#%%
converged_params_list_shaky=get_WT_params(model_type=func,start_guess=params_shaky_list,n_iter=1e5,method="Nelder-Mead",params_dict=params_shaky_dict,custom_settings=[],tol=0.0001)

# %%
