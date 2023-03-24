import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from Models import *
from data_wrangling import *
from itertools import chain
from PyPDF2 import PdfMerger
import inspect
import time
from io import StringIO
import sys
def coef_dict_to_list(coef_dict):
    return list(coef_dict.values())
#the single mutant to be studied
   # function to input paramaters as a list

def get_data_SM(mutation:str):
        df_MT = df
        data = meta_dict["SM"]
        data = data.loc[data['Mutant_ID'] == mutation]
        #WT data missing measurements for inducer = 0.2 so drop last columnn
        data = data[:-1]

        a = re.search("[0-9]",mutation).start()
        mut_col = f"{mutation[:a]}"
        mutation_means = f"{mutation[:a]}_mean"
        df_MT[[mut_col, "Stripe"]] = data[[mutation_means, "Stripe_mean"]].values
        df_MT[["Mutant_ID"]]=mutation
        if mutation.startswith("Output"):
            df_MT["Sensor"]=meta_dict["WT"].Sensor

        return df_MT
#data=pd.read_csv('../data/WT_single.csv')

    #now gonna do the scipy.optimize.minimize

def dict_to_list(params_dict,return_keys=False):
    if return_keys==True:
       a=[list(i.keys()) for i in list(params_dict.values())]
    elif return_keys==False:
        a=[list(i.values()) for i in list(params_dict.values())]

    return list(chain.from_iterable(a))
def coef_dict_to_list(coef_dict):
    return list(coef_dict.values())
#the single mutant to be studied
   # function to input paramaters as a list

def get_data_SM(mutation:str):
        df = pd.read_csv('../data/WT_single.csv')
        df_MT = df
        data = meta_dict["SM"]
        data = data.loc[data['Mutant_ID'] == mutation]
        #WT data missing measurements for inducer = 0.2 so drop last columnn
        data = data[:-1]

        a = re.search("[0-9]",mutation).start()
        mut_col = f"{mutation[:a]}"
        mutation_means = f"{mutation[:a]}_mean"
        df_MT[[mut_col, "Stripe"]] = data[[mutation_means, "Stripe_mean"]].values
        df_MT[["Mutant_ID"]]=mutation
        if mutation.startswith("Output"):
            df_MT["Sensor"]=meta_dict["WT"].Sensor

        return df_MT
#data=pd.read_csv('../data/WT_single.csv')

    #now gonna do the scipy.optimize.minimize
def WT_fit_plot(ax,x, y,params,label:str,data):
        return ax.plot(data[x], y, **params,label=f"{label}")


    #define scatter plotting function with log scales
def WT_Plotter(ax,x,y, params,data):
    out = ax.scatter(data[x], data[y], **params, marker = 'o')
    xScale = ax.set_xscale('log')
    yScale = ax.set_yscale('log')

    return out, xScale, yScale 
# def dict_to_list(params_dict,return_keys=False):
#     if return_keys==True:
#        a=[list(i.keys()) for i in list(params_dict.values())]
#     elif return_keys==False:
#         a=[list(i.values()) for i in list(params_dict.values())]

#     return list(chain.from_iterable(a))

def list_to_dict(old_dict:dict,new_values:list):
    #first check that lengths of lists are compatible
    if len(new_values)==len(dict_to_list(old_dict)):
        i=0
        for key in list(old_dict.keys()):
            for subkey in list(old_dict[key].keys()):
                old_dict[key][subkey]=new_values[i]
                i+=1
    new_dict=old_dict.copy()
    return new_dict


def model_fitting_SM(model_type:callable,params_dict:dict,n_iter:float=1e5,mutant_range:slice=slice(0,len(SM_names)),custom_settings:list=[]):
        #n_iter is how many iterations u want to evaluate
        #mutant_range is if u only want to do a specific few mutants at a time eg for testing
    start_time_all=time.time()    
    df_param_compare = pd.DataFrame(columns=dict_to_list(params_dict,True), index = ['WT'])
    df_param_compare.iloc[0]=dict_to_list(params_dict)
    rss_initial=min_fun(params_list=dict_to_list(params_dict),data=meta_dict['WT'],model_type=model_type)
    df_param_compare['rss']=[rss_initial]
    df_param_compare['time_elapsed_s']=None

    for i in SM_names[mutant_range]: 
        SM_mutant_of_interest=i
        print("Fitting Mutant:",SM_mutant_of_interest)
        start_time_per_mutant=time.time()
        #use model chosen from Models.py to obtain fit of model to data

        df = pd.read_csv('../data/WT_single.csv')

        #substitute mutation data into the WT dataframe for a particular set of mutations
        #for single mutations to start:
        

        #example of above function to get SM dat for Sensor1 mutation:
        S1_df = get_data_SM(SM_mutant_of_interest)

      
        data_ = S1_df

        
        WT_params=dict_to_list(params_dict)
        print("starting paramaters:\n", params_dict)
        
        if SM_mutant_of_interest.startswith("Sensor"):
            
            bnds=generate_bounds(params_dict=params_dict,node="Sensor",custom_settings=custom_settings)[0]
        elif SM_mutant_of_interest.startswith("Regulator"):
            bnds=generate_bounds(params_dict=params_dict,node="Regulator",custom_settings=custom_settings)[0]
        elif SM_mutant_of_interest.startswith("Output"):
            bnds=generate_bounds(params_dict=params_dict,node="Output",custom_settings=custom_settings)[0]
            
        min_result=minimize(min_fun,args=(data_,model_type),x0= WT_params ,method='Nelder-Mead',bounds=bnds,options={"maxiter":n_iter,"disp":True})
        print("finished fitting")

        #plotting the predictions now
        #generating estimates
        Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = model_type(params_list=min_result.x,I_conc=data_.S)

        Sensor_est_array_initial,Regulator_est_array_initial,Output_est_array_initial, Stripe_est_array_initial = model_type(I_conc=data_.S,params_list=WT_params)

        
        #now need to calculate residual sum of logsquares
        #defined as value of minimisation function for converged parameter set
        rss_converged=min_fun(data=data_,params_list=min_result.x,model_type=model_type)
        rss_initial=min_fun(data=data_,params_list=WT_params,model_type=model_type)
        rss_relative=rss_initial/(rss_converged+rss_initial) #% of rss explained by convergence
        
        df_parameters=pd.DataFrame({"epsilon":min_result.x-WT_params,"Initial_guesses":WT_params,"Converged":min_result.x})
        df_parameters.index=dict_to_list(params_dict,return_keys=True)

        
        #add mutant parameter values to df_param_compare to compare
        # a = min_result.x
        # a = np.append(a, rss_converged)
        # dfa = pd.DataFrame(a, index =list(df_param_compare)).T
        # dfa = dfa.set_index(pd.Series([SM_mutant_of_interest]))
        # df_param_compare = pd.concat([df_param_compare, dfa])

        
        fig, ((Sensor, Regulator), (Output, Stripe)) = plt.subplots(2,2, constrained_layout=True)
        
        WT_Plotter(Sensor,"S", "Sensor", {'color':'red'},data=data_)
        WT_fit_plot(Sensor,"S", Sensor_est_array_initial, {'color':'black'},label="Initial Guess",data=data_)
        WT_fit_plot(Sensor,"S", Sensor_est_array,{'color':'red'},label="Converged",data=data_)
        Sensor.set_title(r'inducer -> sensor (GFP output)')
        Sensor.set_xscale('log')
        Sensor.set_yscale('log')
        
        if SM_mutant_of_interest.startswith("Sensor")!=True:
            WT_Plotter(Regulator,"S", "Regulator", {'color': 'blue'},data=data_)
        WT_fit_plot(Regulator,"S", Regulator_est_array, {'color':'blue'},label="Converged",data=data_)
        WT_fit_plot(Regulator,"S", Regulator_est_array_initial, {'color':'black'},label="",data=data_)
        Regulator.set_title(r'inducer ->S -|R (GFP output)')
        Regulator.set_xscale('log')
        Regulator.set_yscale('log')
        

        if SM_mutant_of_interest.startswith("Sensor")!=True:
            WT_Plotter(Output,"S", "Output", {'color': 'purple'},data=data_)
        WT_fit_plot(Output,"S", Output_est_array, {'color':'purple'},label="Converged",data=data_)
        WT_fit_plot(Output,"S" ,Output_est_array_initial, {'color':'black'},label="",data=data_)
        Output.set_title(r'inducer -> S -| Output (GFP)')
        Output.set_xscale('log')
        Output.set_yscale('log')
        

        WT_Plotter(Stripe,"S","Stripe", {'color': 'green'},data=data_)
        WT_fit_plot(Stripe,"S", Stripe_est_array, {'color':'green'},label="Converged",data=data_)
        WT_fit_plot(Stripe,"S", Stripe_est_array_initial, {'color':'black'},label="",data=data_)
        Stripe.set_title(r'Full circuit with stripe')
        fig.legend(bbox_to_anchor=(1.3, 1))
        title = ["SM data type data plots for mutation", SM_mutant_of_interest,"using model:",str(model_type.__qualname__)]
        txt=f''' Across all four plots: \n
        RSS (converged)={round(rss_converged,3)} \n
        RSS (initial)={round(rss_initial,3)}\n
        RSS (% reduction)={round(rss_relative,3)}\n
        '''
        txt+=str(df_parameters)
        fig.text(.11,-.81,txt,wrap=True)
        txt=f'''{min_result}
        '''
        fig.text(0.7,-.81,txt,wrap=True)
        old_stdout=sys.stdout
        result=StringIO()
        sys.stdout=result
        print("--- %s seconds ---" % (time.time() - start_time_per_mutant))
        
        txt="time elapsed for this fit \n"+result.getvalue()
        fig.text(1.1,0.5,txt)
        sys.stdout=old_stdout
        txt=str(bnds)
        fig.text(1.1,0,txt)
        plt.suptitle(title,fontweight="bold")

        plt.show()
        print("final parameter estimates:", min_result.x)


        print("done fitting, now exporting")
        fig.savefig(os.path.join("..","results",f"{SM_mutant_of_interest}"+"_"+str(model_type.__qualname__+".pdf")), bbox_inches='tight')
        

        time_elapsed=str("%s" % (time.time()-start_time_per_mutant))
        a=list(df_parameters.transpose().iloc[2]) #collecting converged parameters for export
        a.append(rss_converged)
        a.append(time_elapsed)
        df_param_compare.loc[f'{SM_mutant_of_interest}']=a
        print(i,df_param_compare)

        print("time elapsed for this fit \n",)
        print("--- %s seconds ---" % (time.time() - start_time_per_mutant))
       
        # %%
    df_param_compare.to_excel(os.path.join("..","data",f"{model_type.__qualname__}"+"SM_params.xlsx"))
       

    pdfs= [os.path.join("..","results",(s+"_"+str(model_type.__qualname__)+".pdf")) for s in SM_names]
    merger = PdfMerger()

    for pdf in pdfs:
        merger.append(pdf)

    merger.write(os.path.join("..","results","SM_models_"+str(model_type.__qualname__+".pdf")))
    merger.close()
    print("done with all, time elapsed: \n")
    print("--- %s seconds ---" % (time.time() - start_time_all))
    return 1

#%%
def get_WT_params(model_type,start_guess:list,params_dict:dict,custom_settings:list,tol:float,method="Nelder-Mead",n_iter:float=1e5,node=""):
    #this function will estimate the wild type parameters for a given model.
    #now loading wt dataframe
    start_time=time.time()   
    if start_guess==[]:
        start_guess=[1]*20
    data_=meta_dict['WT']
    bnds=generate_bounds(params_dict=params_dict,node=node,custom_settings=custom_settings)[0]
    min_result=minimize(min_fun,args=(data_,model_type),x0=start_guess,method='Nelder-Mead',tol=1,bounds=bnds,options={"maxiter":n_iter,"disp":True})

    Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = model_type(params_list=min_result.x,I_conc=data_.S)
    
    Sensor_est_array_initial,Regulator_est_array_initial,Output_est_array_initial, Stripe_est_array_initial = model_type(I_conc=data_.S,params_list=start_guess)

    rss_converged=min_fun(data=data_,params_list=min_result.x,model_type=model_type)
    rss_initial=min_fun(data=data_,params_list=start_guess,model_type=model_type)
    rss_relative=rss_initial/(rss_converged+rss_initial) #% of rss explained by 
  
    fig, ((Sensor, Regulator), (Output, Stripe)) = plt.subplots(2,2, constrained_layout=True)
    
    WT_Plotter(Sensor,"S", "Sensor", {'color':'red'},data=data_)
    WT_fit_plot(Sensor,"S", Sensor_est_array_initial, {'color':'black'},label="Initial Guess",data=data_)
    WT_fit_plot(Sensor,"S",Sensor_est_array,{'color':'red'},label="Converged",data=data_)
    Sensor.set_title(r'inducer -> sensor (GFP output)')
    Sensor.set_xscale('log')
    Sensor.set_yscale('log')
    
    
    WT_Plotter(Regulator,"S", "Regulator", {'color': 'blue'},data=data_)
    WT_fit_plot(Regulator,"S", Regulator_est_array, {'color':'blue'},label="Converged",data=data_)
    WT_fit_plot(Regulator,'S', Regulator_est_array_initial, {'color':'black'},label="",data=data_)
    Regulator.set_title(r'inducer ->S -|R (GFP output)')
    Regulator.set_xscale('log')
    Regulator.set_yscale('log')
    

    WT_Plotter(Output,"S", "Output", {'color': 'purple'},data=data_)
    WT_fit_plot(Output,"S", Output_est_array, {'color':'purple'},label="Converged",data=data_)
    WT_fit_plot(Output,"S", Output_est_array_initial, {'color':'black'},label="",data=data_)
    Output.set_title(r'inducer -> S -| Output (GFP)')
    Output.set_xscale('log')
    Output.set_yscale('log')
    

    WT_Plotter(Stripe,"S","Stripe", {'color': 'green'},data=data_)
    WT_fit_plot(Stripe,"S", Stripe_est_array, {'color':'green'},label="Converged",data=data_)
    WT_fit_plot(Stripe,"S", Stripe_est_array_initial, {'color':'black'},label="",data=data_)
    Stripe.set_title(r'Full circuit with stripe')
    fig.legend(bbox_to_anchor=(1.3, 1))
    title = ["WT data plots ","using model:",str(model_type.__qualname__)]
    df_parameters=pd.DataFrame({"epsilon":min_result.x-start_guess,"Initial_guesses":start_guess,"Converged":min_result.x})
    df_parameters.index=dict_to_list(params_dict,return_keys=True)

    txt=f''' Across all four plots: \n
    RSS (converged)={round(rss_converged,3)} \n
    RSS (initial)={round(rss_initial,3)}\n
    RSS (% reduction)={round(rss_relative,3)}\n
    '''
    txt+=str(df_parameters)
    fig.text(.11,-.81,txt,wrap=True)
    txt=f'''{min_result}
    '''
    fig.text(0.7,-.81,txt,wrap=True)
    txt=str(bnds)
    fig.text(1.1,0,txt)

    old_stdout=sys.stdout
    result=StringIO()
    sys.stdout=result
    print("--- %s seconds ---" % (time.time() - start_time))
    
    txt="time elapsed for this fit \n"+result.getvalue()
    fig.text(1.1,0.5,txt)
    sys.stdout=old_stdout


    plt.suptitle(title,fontweight="bold")

    plt.show()
    print("final parameter estimates:", min_result.x)

    
    return min_result.x
#%%
#dictionary structure for input to model
#to add, feature that will generate initial guess from wildtype
#which u then feed to the actual running of the function
#%%
def generate_bounds(params_dict:dict,node:str="",custom_settings:list=[]):
    #need to generate a tuple
    #where all common variables are allowed to vary?
    #use custom settings to set
    #custom settings can be a list of sub lists
    #first sublist is lower bound
    #second is upper
    #third is string to identify
    #to have bound same as wildtype, make the value equal to "same"
    
    #we will do this by generating the values in a list
    #default behaviour when running for first time should be to make all bounds greater than one.
    lower=0
        
    WT_params=dict_to_list(params_dict)
    param_names=dict_to_list(params_dict,True)
    bnds_list=[None]*len(WT_params)
    for i in range(0,len(WT_params)):
        bnds_list[i]=[lower,None]
    
    #now we want to apply some changes depending on nodes and custom settings etc
    if node=="Sensor":
        #first set everything to be fixed, then unfix sensor

        for i in range(0,len(bnds_list)):
            bnds_list[i]=[WT_params[i],WT_params[i]]
        bnds_list

        #sensor keys
        keys=list(params_dict["sen_params"].keys())
        #sensor values
        values=params_dict["sen_params"].values()
        #indices of sensor's keys
        keys_indices_to_change= [i for i,x in enumerate(param_names) if x in keys]
        #indices
        keys_values_to_keep_constant=[WT_params[i] for i in keys_indices_to_change]
        for i in range(0,len(keys_indices_to_change)):
            
            bnds_list[keys_indices_to_change[i]]=[0,None]
        #this changes sensor nodes to freely change
        #now we need to set all other nodes to be fixed

    elif node=="Regulator":
        #first set everything to be fixed, then unfix regulator
        
        for i in range(0,len(bnds_list)):
            bnds_list[i]=[WT_params[i],WT_params[i]]
   
        
        keys=list(params_dict["reg_params"].keys())
        values=params_dict["reg_params"].values()

        keys_indices_to_change= [i for i,x in enumerate(param_names) if x in keys]
        keys_values_to_keep_constant=[WT_params[i] for i in keys_indices_to_change]
        for i in range(0,len(keys_indices_to_change)):
            
            bnds_list[keys_indices_to_change[i]]=[0,None]
    elif node=="Output":
        #first set everything to be fixed, then unfix sensor
        
        for i in range(0,len(bnds_list)):
            bnds_list[i]=[WT_params[i],WT_params[i]]
        
        keys=list(params_dict["out_params"].keys())
        values=params_dict["out_params"].values()

        keys_indices_to_change= [i for i,x in enumerate(param_names) if x in keys]
        keys_values_to_keep_constant=[WT_params[i] for i in keys_indices_to_change]
        for i in range(0,len(keys_indices_to_change)):
            
            bnds_list[keys_indices_to_change[i]]=[0,None]
    elif node=="Output_half":
        #first set everything to be fixed, then unfix sensor
        
        for i in range(0,len(bnds_list)):
            bnds_list[i]=[WT_params[i],WT_params[i]]
        
        keys=list(params_dict["out_h_params"].keys())
        values=params_dict["out_h_params"].values()

        keys_indices_to_change= [i for i,x in enumerate(param_names) if x in keys]
        keys_values_to_keep_constant=[WT_params[i] for i in keys_indices_to_change]
        for i in range(0,len(keys_indices_to_change)):
            
            bnds_list[keys_indices_to_change[i]]=[0,None]
    
    #now adding what to do for free params
    keys=list(params_dict['free_params'].keys())
    keys_indices_to_change=[i for i,x in enumerate(param_names) if x in keys]
    for i in range(0,len(keys_indices_to_change)):
            
            bnds_list[keys_indices_to_change[i]]=[lower,None]
    
    
    #after applying which bounds are constant we now want to apply the custom settings
    if custom_settings!=[]:
        low=custom_settings[0]
        high=custom_settings[1]
        parameters=custom_settings[2]
        #custom_settings=[[0,0,0],[1,1,1],["Co","No","Ao"]]
        #first find the parameter to change
        index_to_change=[i for i,x in enumerate(param_names) if x in parameters]
        for i in range(0,len(index_to_change)):
            bnds_list[index_to_change[i]]=[low[i],high[i]]


    for i in range(0,len(bnds_list)):
        bnds_list[i]=tuple(bnds_list[i])
    bnds_tuple=tuple(bnds_list)
    return bnds_tuple,param_names
    #now applying custom settings
    #param_names
# %%
def Plotter(model_type,start_guess:list,params_dict:dict,custom_settings:list,tol:float,method="Nelder-Mead",n_iter:float=1e5,node=""):
    #this function will estimate the wild type parameters for a given model.
    #now loading wt dataframe
    start_time=time.time()   
    # if start_guess==[]:
    #     start_guess=[1]*20
    data_=meta_dict['WT']
    bnds=generate_bounds(params_dict=params_dict,node=node,custom_settings=custom_settings)[0]
    #min_result=minimize(min_fun,args=(data_,model_type),x0=start_guess,method='Nelder-Mead',tol=1,bounds=bnds,options={"maxiter":n_iter,"disp":True})

    #Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = #model_type(params_list=min_result.x,I_conc=data_.S)
    
    Sensor_est_array_initial,Regulator_est_array_initial,Output_est_array_initial, Stripe_est_array_initial = model_type(I_conc=data_.S,params_list=start_guess)

    #rss_converged=min_fun(data=data_,params_list=min_result.x,model_type=model_type)
    #rss_initial=min_fun(data=data_,params_list=start_guess,model_type=model_type)
    #rss_relative=rss_initial/(rss_converged+rss_initial) #% of rss explained by 
  
    fig, ((Sensor, Regulator), (Output, Stripe)) = plt.subplots(2,2, constrained_layout=True)
    
    WT_Plotter(Sensor,"S", "Sensor", {'color':'red'},data=data_)
    WT_fit_plot(Sensor,"S", Sensor_est_array_initial, {'color':'black'},label="Initial Guess",data=data_)
    #WT_fit_plot(Sensor,"S",Sensor_est_array,{'color':'red'},label="Converged",data=data_)
    Sensor.set_title(r'inducer -> sensor (GFP output)')
    Sensor.set_xscale('log')
    Sensor.set_yscale('log')
    
    
    WT_Plotter(Regulator,"S", "Regulator", {'color': 'blue'},data=data_)
    #WT_fit_plot(Regulator,"S", Regulator_est_array, {'color':'blue'},label="Converged",data=data_)
    WT_fit_plot(Regulator,'S', Regulator_est_array_initial, {'color':'black'},label="",data=data_)
    Regulator.set_title(r'inducer ->S -|R (GFP output)')
    Regulator.set_xscale('log')
    Regulator.set_yscale('log')
    

    WT_Plotter(Output,"S", "Output", {'color': 'purple'},data=data_)
    #WT_fit_plot(Output,"S", Output_est_array, {'color':'purple'},label="Converged",data=data_)
    WT_fit_plot(Output,"S", Output_est_array_initial, {'color':'black'},label="",data=data_)
    Output.set_title(r'inducer -> S -| Output (GFP)')
    Output.set_xscale('log')
    Output.set_yscale('log')
    

    WT_Plotter(Stripe,"S","Stripe", {'color': 'green'},data=data_)
    #WT_fit_plot(Stripe,"S", Stripe_est_array, {'color':'green'},label="Converged",data=data_)
    WT_fit_plot(Stripe,"S", Stripe_est_array_initial, {'color':'black'},label="",data=data_)
    Stripe.set_title(r'Full circuit with stripe')
    fig.legend(bbox_to_anchor=(1.3, 1))
    title = ["WT data plots ","using model:",str(model_type.__qualname__)]
    #df_parameters=pd.DataFrame({"epsilon":min_result.x-start_guess,"Initial_guesses":start_guess,"Converged":min_result.x})
    #df_parameters.index=dict_to_list(params_dict,return_keys=True)
    
    #txt=f''' Across all four plots: \n
    #RSS (converged)={round(rss_converged,3)} \n
    #RSS (initial)={round(rss_initial,3)}\n
    #RSS (% reduction)={round(rss_relative,3)}\n
    #'''
    ##txt+=str(df_parameters)
    #fig.text(.11,-.81,txt,wrap=True)
    #txt=f'''{min_result}
    #'''
    # #fig.text(0.7,-.81,txt,wrap=True)
    # txt=str(bnds)
    # fig.text(1.1,0,txt)

    # old_stdout=sys.stdout
    # result=StringIO()
    # sys.stdout=result
    # print("--- %s seconds ---" % (time.time() - start_time))
    
    # txt="time elapsed for this fit \n"+result.getvalue()
    # fig.text(1.1,0.5,txt)
    # sys.stdout=old_stdout


    plt.suptitle(title,fontweight="bold")

    plt.show()
    print("final parameter estimates:")
    print(start_guess)

    
    return 