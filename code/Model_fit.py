import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from Models import *
from data_wrangling import *
from numpy import sin
from numpy import sqrt
from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot
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
#Hill
params_dict_hill={"sen_params":{"A_s":1,"B_s":1,"C_s":1,"N_s":1},"reg_params":{"A_r":1,"B_r":1,"C_r":1,"N_r":1},"out_h_params":{"A_h":1,"B_h":1,"C_h":1},"out_params":{"A_o":1,"B_o":1,"C_o":1,"N_o":1},"free_params":{"F_o":1}}

# params_list_hill=[618.05, 16278.86, 1300.65, 1.23445789e+00,2.19715875e+03, 5.09396044e+04, 8.76632583e-03, 1.37272209e+00, 3.97046404e+03, 2.81037530e+04, 5.99908397e-04, 8.61568305e-01, 7.03425130e-01, 7.57153375e+00, 1.25692066e+00, 3.39280741e+00]

params_list_hill=dict_to_list(params_dict_hill)

# params_list_hill=[618.05, 16278.86, 1300.65, 1.23445789e+00,2.19715875e+03, 5.09396044e+04, 8.76632583e-03, 1.37272209e+00, 3.97046404e+03, 2.81037530e+04, 5.99908397e-04, 8.61568305e-01, 7.03425130e-01, 7.57153375e+00, 1.25692066e+00, 3.39280741e+00]

I_conc=meta_dict["WT"].S
hill=model_hill(params_list_hill,I_conc)
func=model_hill.model

converged_params_list_hill=get_WT_params(model_type=func,start_guess=params_list_hill,n_iter=1e5,method="Nelder-Mead",params_dict=params_dict_hill,custom_settings=[],tol=0.0001)

#%%
#Thermodynamic
therm_model=model_thermodynamic(params_list=[1]*14,I_conc=meta_dict["WT"].S)
func=therm_model.model
params_therm_dict={"sen_params":{"P_b":6.70063321e-02,"P_u":9.26794143e-07,"K_12":2.23007522e+03,"C_pa":1.84833176e+09,"A_s":1.45247730e+04},"reg_params":{"P_r":1.67754325e+00,"C_pt":1.26659501e-02,"K_t":6.16701374e-05,"A_r":1.09634721e+05},"out_h_params":{},"out_params":{"P_o":1.67754325e+00,"C_pl":0.00575464637, "K_l":2.40059869e-06,"A_o":7.98011665e+04},"free_params":{},"fixed_params":{"F_o":1.48194306e+00}}
params_therm_list=dict_to_list(params_therm_dict)
params_therm_list=get_WT_params(model_type=func,start_guess=params_therm_list,n_iter=1e5,method="Nelder-Mead",params_dict=params_therm_dict,custom_settings=[[1,0,0,0,0],[None,1,1,None,None],["C_pa","C_pt","C_pl","P_p","F_o"]],tol=1)

#%%
# params_shaky_dict= {"sen_params":{"A_s":6.50714912e+021,"B_s":1.62599799e+04,"C_s":1.29644889e+03,"N_s":1.15406707e+00},"reg_params":{"A_r":2.02001922e+03,"B_r":2.36888092e+04,"C_r":1.03576041e-02,"N_r":9.10072254e-01},"out_h_params":{"A_h":1.43802212e+02,"B_h":5.02382714e+04,"C_h":9.29129276e-04},"out_params":{"A_o":1.67389421e+00,"B_o":8.95341925e-01,"C_o":2.65769862e+00,"N_o":1.37995266e+00},"free_params":{"F_o":2.36128375e+00}}

# params_shaky_dict= {"sen_params":{"A_s":4.61637106e+03,"B_s":1.85909749e+04,"C_s":1.25108743e+03,"N_s":1.19122010e+00},"reg_params":{"A_r":2.02001922e+03,"B_r":2.36888092e+04,"C_r":1.03576041e-02,"N_r":9.10072254e-01},"out_h_params":{"A_h":1.43802212e+02,"B_h":5.02382714e+04,"C_h":9.29129276e-04},"out_params":{"A_o":1.67389421e+00,"B_o":8.95341925e-01,"C_o":2.65769862e+00,"N_o":1.37995266e+00},"free_params":{"F_o":2.36128375e+00}}

# params_shaky_dict= {"sen_params":{"A_s":6.363e+02,"B_s":2.70e+04,"C_s":1.029e+03,"N_s":1.097e+00},"reg_params":{"A_r":5.403e+01,"B_r":5.169e+02,"C_r":1.958e-02,"N_r":3.582e-01},"out_h_params":{"A_h":1.4380e+02,"B_h":5.0234e+04,"C_h":9.29176e-04},"out_params":{"A_o":2.2e-04,"B_o":8e+03,"C_o":2.19e-04,"N_o":5.45},"free_params":{"F_o":0.5}}
# params_shaky_dict= {"sen_params":{"A_s":659,"B_s":25299,"C_s":1249,"N_s":1.19},"reg_params":{"A_r":2118,"B_r":410,"C_r":0.13,"N_r":1.1},"out_h_params":{"A_h":1.43802212e+02,"B_h":5.02382714e+04,"C_h":9.29129276e-04},"out_params":{"A_o":0.00022,"B_o":8000,"C_o":0.000219,"N_o":5.45},"free_params":{"F_o":0.5}}
# params_shaky_dict= {"sen_params":{"A_s":7.27203090e+02,"B_s":2.53693912e+04,"C_s":1.29142360e+03,"N_s":1.26521014e+00},"reg_params":{"A_r":2.22675722e+03,"B_r":3.80626639e+04,"C_r":1.10091516e+00,"N_r":1.43045364e+00},"out_h_params":{"A_h":3.97046404e+03,"B_h":2.81037530e+04,"C_h":5.99908397e-04},"out_params":{"A_o":5.38514322e+00,"B_o":9.44080098e-01,"C_o":2.34635713e+04,"N_o":4.62788051e+01},"free_params":{"F_o":3.36238128e+02}}
# params_shaky_dict= {"sen_params":{"A_s":7.27203090e+02,"B_s":2.53693912e+04,"C_s":1.29142360e+03,"N_s":1.26521014e+00},"reg_params":{"A_r":2.22675722e+03,"B_r":3.80626639e+04,"C_r":1.10091516e+00,"N_r":1.43045364e+00},"out_h_params":{},"out_params":{"A_o":5.38514322e+00,"B_o":9.44080098e-01,"C_o":2.34635713e+04,"N_o":4.62788051e+01},"free_params":{"F_o":3.36238128e+02}}

params_shaky_dict= {"sen_params":{"A_s":6.59203090e+02,"B_s":2.52693912e+04,"C_s":1.29142360e+03,"N_s":1.26521014e+00},"reg_params":{"A_r":2.22675722e+03,"B_r":3.80626639e+04,"C_r":1.10091516e+00,"N_r":1.43045364e+00},"out_h_params":{},"out_params":{"A_o":2.43045364e-04,"B_o":9.579995752e+03,"C_o":2.29e-04,"N_o":3.356268e+00},"free_params":{"F_o":3.36238128e-01}}

params_shaky_list=dict_to_list(params_shaky_dict)

shaky_model= model_hill_shaky(params_list=params_shaky_list, I_conc=meta_dict["WT"].S)
func = shaky_model.model

converged_params_list_shaky=get_WT_params(model_type=func,start_guess=params_shaky_list,n_iter=1e5,method="Nelder-Mead",params_dict=params_shaky_dict,custom_settings=[],tol=0.0001)


#%%

data = meta_dict['WT']
#generate x and y data for sensor
#provide initial values and then use minimise and curvefit to try and get good parameters, move onto inducer, output
#meaning of the parameters, correlation between two


# %%
params_shaky_dict= {"sen_params":{"A_s":659,"B_s":25299,"C_s":1249,"N_s":1.19},"reg_params":{"A_r":2118,"B_r":410,"C_r":0.13,"N_r":1.1},"out_h_params":{"A_h":1.43802212e+02,"B_h":5.02382714e+04,"C_h":9.29129276e-04},"out_params":{"A_o":0.00022,"B_o":8000,"C_o":0.000219,"N_o":5.45},"free_params":{"F_o":0.5}}
