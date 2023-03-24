#From source data, outputs a disctionary with 4 keys (WT,SM,DM,TM)
#%%
import os
import math
import pandas as pd
import numpy as np
import re
#%%
meta_dict={"WT":None,"SM":None,"DM":None,"TM":None}
meta_dict["WT"]=pd.read_csv('../data/WT_single.csv') #wildtype fluorescence for each node over I_conc
#SM
path_main= "../data/mutants_separated"
Stripe_data = pd.ExcelFile('../data/Source_Data.xlsx')
Stripe_data.sheet_names #source data has multiple sub-sheets
SM_stripes = pd.read_excel(Stripe_data, 'Figure 1')
#SM_stripes['Sensor 1']
all_files=os.listdir(os.path.join(path_main,"mutants_separated"))
dat_files=[x for x in all_files if '.dat' in x]
SM_names=[x[:-4] for x in dat_files] #get all single mutant names, but NOT SORTED

mean, stdev, Output_mean, Output_stdev, Regulator_mean, Regulator_stdev, Sensor_mean, Sensor_stdev, inducer, genotype, Stripe_mean, Stripe_stdev = [],[],[],[],[],[],[],[],[],[],[],[]

for mutant in SM_names:
    mean=[]
    stdev=[]

    a = re.search("[0-9]",mutant).start() #index of mutant identifier
    mutant_stripe = f"{mutant[:a]} {mutant[a:]}"
    
    mutant_table=pd.read_table(os.path.join(path_main,"mutants_separated",(mutant+".dat")),header=None)
    indexes=int(len(mutant_table)/3)
    ind = mutant_table.iloc[0:indexes,0] #concentrations
    genotype += [mutant]*indexes
    inducer.extend(ind)
    for i in range(0,indexes):       
        mean.append(mutant_table.iloc[[i,i+indexes,i+(indexes*2)],1].mean())
        stdev.append(mutant_table.iloc[[i,i+indexes,i+indexes*2],1].std())

    Stripe_mean += list(SM_stripes[mutant_stripe][1:indexes+1])
    std_col = list(SM_stripes.columns).index(mutant_stripe)+1
    Stripe_stdev += list(SM_stripes.iloc[1:indexes+1, std_col])

    if mutant.startswith("Output"):

        Output_mean.extend(mean)        
        Output_stdev.extend(stdev)

        Regulator_mean.extend([math.nan]*len(mean))
        Regulator_stdev.extend([math.nan]*len(mean))
        Sensor_mean.extend([math.nan]*len(mean))
        Sensor_stdev.extend([math.nan]*len(mean))

    if mutant.startswith("Regulator"):

        Regulator_mean.extend(mean)
        Regulator_stdev.extend(stdev)

        Output_mean.extend([math.nan]*len(mean))
        Output_stdev.extend([math.nan]*len(mean))
        Sensor_mean.extend([math.nan]*len(mean))
        Sensor_stdev.extend([math.nan]*len(mean))
        
    if mutant.startswith("Sensor"):

        Sensor_mean.extend(mean)
        Sensor_stdev.extend(stdev)

        Output_mean.extend([math.nan]*len(mean))
        Output_stdev.extend([math.nan]*len(mean))
        Regulator_mean.extend([math.nan]*len(mean))
        Regulator_stdev.extend([math.nan]*len(mean))

#Missing data for S7 and O7 @ I_conc 0.00020. Replaced with NaN
sm_df = pd.DataFrame({"Inducer" :  inducer,"Mutant_ID": genotype,'Output_mean': Output_mean, 'Output_stdev': Output_stdev, 'Regulator_mean': Regulator_mean,"Regulator_stdev": Regulator_stdev,"Sensor_mean": Sensor_mean,"Sensor_stdev": Sensor_stdev,"Stripe_mean": Stripe_mean,"Stripe_stdev": Stripe_stdev})

meta_dict["SM"] = sm_df

#now read in the double mutant data, only collected for low, medium, high inducer concs.
#low = 0, medium = 0.0002, high = 0.2
Stripe_data = pd.ExcelFile('../data/Source_Data.xlsx')
Stripe_data.sheet_names
stripes = pd.read_excel(Stripe_data, 'Figure 2', header = 1, usecols="A:E")
DM_stripes = stripes[(stripes['genotype category'] == "pairwise") | (stripes['genotype'] == "WT")]
#triple mutants
TM_stripes = stripes[(stripes['genotype category'] == "triple") | (stripes['genotype'] == "WT")]

meta_dict["DM"] = DM_stripes
meta_dict["TM"] = TM_stripes

# %%
