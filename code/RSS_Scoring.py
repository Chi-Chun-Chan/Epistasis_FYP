#%%
from Model_fitting_functions import *
from Models import *
from Models import model_hill
from scipy.optimize import minimize
#%%
#calculates RSS
def min_func_all(params_list,data,model_type):
        log_sen = np.log10(data.Sensor)
        log_reg = np.log10(data.Regulator)
        log_out = np.log10(data.Output)
        log_stripe = np.log10(data.Stripe)   
        ind = data.S
        Sensor_est, Regulator_est, Output_est, Stripe_est = model_type(params_list,I_conc=ind)

        #put back the sensor/stripe minimisation
        log_sen_est = np.log10(Sensor_est)
        log_reg_est = np.log10(Regulator_est)
        log_out_est = np.log10(Output_est)
        log_stripe_est = np.log10(Stripe_est)

        if "Mutant_ID" in data:
            mutant_id=data.Mutant_ID[0]
            if mutant_id.startswith("Sensor"):
                #need to ignore reg and output in fitting
                log_reg,log_reg_est,log_out,log_out_est=0,0,0,0

        result = np.power((log_sen - log_sen_est), 2)
        result += np.power((log_reg - log_reg_est), 2)
        result += np.power((log_out - log_out_est), 2)
        result += np.power((log_stripe - log_stripe_est), 2)
        return np.sum(result) 

#%%
def RSS_Score(param_list:list,model_type,data_):
        Model = model_type(params_list=[1]*13, I_conc=meta_dict["WT"].S)
        func = Model.model
        rss_converged=min_func_all(data=data_,params_list=param_list,model_type=func)
        return rss_converged
        
# %%
