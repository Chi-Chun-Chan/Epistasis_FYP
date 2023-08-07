''' Epistasis summary python file'''
#%%
'''Import functions from prior python(very messy apologies)'''
from Hill_Model_Fit import *
#%%
'''Extracts 10000 parameters for each combined mutant, store as seperate files (already done)'''
def pairwise_params_to_dfs(prior_mutant):
    '''Get all the parameters'''
    size = 100 #number of random wildtype parameters
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
        #get genotypeID
        mutant_list = get_mut_names(genotypes)
        mut1 = mutant_list[0]
        mut2 = mutant_list[1]


        parameters = pd.DataFrame({
                        "A_s":[],
                        "B_s":[],
                        "C_s":[],
                        "N_s":[],
                        "MA_s":[],
                        "MB_s":[],
                        "MC_s":[],
                        "MN_s":[], 
                        "A_r":[],
                        "B_r":[],
                        "C_r":[],
                        "N_r":[],
                        "MA_r":[],
                        "MB_r":[],
                        "MC_r":[],
                        "MN_r":[],
                        "A_o":[],
                        "B_o":[],
                        "C_o":[],
                        "N_o":[],
                        "F_o":[],
                        "MA_o":[],
                        "MB_o":[],
                        "MC_o":[],
                        "MN_o":[],
                        "MF_o":[],
                            })
        for i in range(0,size):
            WT_pars, Mut1_pars_array, Mut2_pars_array = get_combo_params(mutant_list)
            for Mut1_pars,Mut2_pars in zip(Mut1_pars_array,Mut2_pars_array):
            #identification of mutant types
                if mut1.startswith('Sensor') & mut2.startswith('Output'):
                    M = {'As':Mut1_pars[0],'Bs':Mut1_pars[1],'Cs':Mut1_pars[2],'Ns':Mut1_pars[3],'Ar':0.0,'Br':0.0,'Cr':0.0,'Nr':0.0,'Ao':Mut2_pars[0],'Bo':Mut2_pars[1],'Co':Mut2_pars[2],'No':Mut2_pars[3],'Fo':0.0}
                elif mut1.startswith('Sensor') & mut2.startswith('Regulator'):
                    M = {'As':Mut1_pars[0],'Bs':Mut1_pars[1],'Cs':Mut1_pars[2],'Ns':Mut1_pars[3],'Ar':Mut2_pars[0],'Br':Mut2_pars[1],'Cr':Mut2_pars[2],'Nr':Mut2_pars[3],'Ao':0.0,'Bo':0.0,'Co':0.0,'No':0.0,'Fo':0.0}
                elif mut1.startswith('Regulator') & mut2.startswith('Output'):
                    M = {'As':0.0,'Bs':0.0,'Cs':0.0,'Ns':0.0,'Ar':Mut1_pars[0],'Br':Mut1_pars[1],'Cr':Mut1_pars[2],'Nr':Mut1_pars[3],'Ao':Mut2_pars[0],'Bo':Mut2_pars[1],'Co':Mut2_pars[2],'No':Mut2_pars[3],'Fo':0.0}
                elif mut1.startswith('Regulator') & mut2.startswith('Sensor'):
                    M = {'As':Mut2_pars[0],'Bs':Mut2_pars[1],'Cs':Mut2_pars[2],'Ns':Mut2_pars[3],'Ar':Mut1_pars[1],'Br':Mut1_pars[1],'Cr':Mut1_pars[1],'Nr':Mut1_pars[1],'Ao':0.0,'Bo':0.0,'Co':0.0,'No':0.0,'Fo':0.0}
                elif mut1.startswith('Output') & mut2.startswith('Regulator'):
                    M = {'As':0.0,'Bs':0.0,'Cs':0.0,'Ns':0.0,'Ar':Mut2_pars[0],'Br':Mut2_pars[1],'Cr':Mut2_pars[2],'Nr':Mut2_pars[3],'Ao':Mut1_pars[0],'Bo':Mut1_pars[1],'Co':Mut1_pars[2],'No':Mut1_pars[3],'Fo':0.0}
                elif mut1.startswith('Output') & mut2.startswith('Sensor'):
                    M = {'As':Mut2_pars[0],'Bs':Mut2_pars[1],'Cs':Mut2_pars[2],'Ns':Mut2_pars[3],'Ar':0.0,'Br':0.0,'Cr':0.0,'Nr':0.0,'Ao':Mut1_pars[0],'Bo':Mut1_pars[1],'Co':Mut1_pars[2],'No':Mut1_pars[3],'Fo':0.0}
                else:
                    raise KeyError('Mutant names invalid')
            
                par_dict = {
                    "A_s":10**WT_pars[0],
                    "B_s":10**WT_pars[1],
                    "C_s":10**WT_pars[2],
                    "N_s":10**WT_pars[3], #All WT params were logged to find conditional mutants, this is to convert them back into non-logged values
                    "MA_s":10**M['As'],
                    "MB_s":10**M['Bs'],
                    "MC_s":10**M['Cs'],
                    "MN_s":10**M['Ns'], 
                    "A_r":10**WT_pars[4],
                    "B_r":10**WT_pars[5],
                    "C_r":10**WT_pars[6],
                    "N_r":10**WT_pars[7],
                    "MA_r":10**M['Ar'],
                    "MB_r":10**M['Br'],
                    "MC_r":10**M['Cr'],
                    "MN_r":10**M['Nr'],
                    "A_o":10**WT_pars[8],
                    "B_o":10**WT_pars[9],
                    "C_o":10**WT_pars[10],
                    "N_o":10**WT_pars[11],
                    "F_o":10**WT_pars[12],
                    "MA_o":10**M['Ao'],
                    "MB_o":10**M['Bo'],
                    "MC_o":10**M['Co'],
                    "MN_o":10**M['No'],
                    "MF_o":10**M['Fo'],
                        }
            
                par_list = list(par_dict.values())

                parameters.loc[len(parameters)] = par_list
        
        path = f'../results/Pairwise_params/{genotypes}.csv'
        count = count + 1
        parameters.to_csv(path, index = False)
        print('mutant ', genotypes, 'completed, ', count, 'out of 300')
def triplet_params_to_dfs(prior_mutant):
    size = 100 #number of random wildtype parameters
    TM_names = TM_stripes['genotype'].tolist()
    TM_names = list(set(TM_names[3:]))
    TM_names.sort()

    if prior_mutant == None:
        mutant_range:slice=slice(0,len(TM_names))
        count = 0
    else:
        position = TM_names.index(prior_mutant)
        count = position
        mutant_range:slice=slice(position+1,len(TM_names))

    for genotypes in TM_names[mutant_range]:
        #get genotypeID
        mutant_list = get_mut_names(genotypes)
        mut1 = mutant_list[0]
        mut2 = mutant_list[1]

        rndint = np.random.randint(low=0, high=1e7)
        timeseed = time.time_ns() % 2**16
        np.random.seed(rndint+timeseed)
        seed(rndint+timeseed)

        #plot pairwise fit
        #selects mutant shortcode and assembles into correct mutant ID

        TM_df = meta_dict['TM']

        #All mutants are in order of R_S_O

        mutid = get_mut_ids(mutant_list)

        trip_mut_dict = TM_df[TM_df['genotype'] == mutid]
        

        tripwise_fluo = []
        for fluo in trip_mut_dict['obs_fluo_mean']:
            tripwise_fluo.append(fluo)


        tripwise_inducer = [0.00001, 0.0002, 0.2]

        ind = pd.DataFrame(tripwise_inducer)
        #plot mutant fits
        hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)

        WT_pars_array = np.empty(shape=(size,13))
        Mut1_pars_array = np.empty(shape=(size,4))
        Mut2_pars_array = np.empty(shape=(size,4))
        Mut3_pars_array = np.empty(shape=(size,4))

        low = []
        med = []
        high = []
        parameters = pd.DataFrame({
                "A_s":[],
                "B_s":[],
                "C_s":[],
                "N_s":[],
                "MA_s":[],
                "MB_s":[],
                "MC_s":[],
                "MN_s":[], 
                "A_r":[],
                "B_r":[],
                "C_r":[],
                "N_r":[],
                "MA_r":[],
                "MB_r":[],
                "MC_r":[],
                "MN_r":[],
                "A_o":[],
                "B_o":[],
                "C_o":[],
                "N_o":[],
                "F_o":[],
                "MA_o":[],
                "MB_o":[],
                "MC_o":[],
                "MN_o":[],
                "MF_o":[],
                    })

        for i in range(0,size):
            WT_pars, Mut1_pars_array, Mut2_pars_array, Mut3_pars_array = get_combo_params_TM(mutant_list)
            for Mut1_pars,Mut2_pars,Mut3_pars in zip(Mut1_pars_array,Mut2_pars_array, Mut3_pars_array):
            #identification of mutant types
                if mut1.startswith('Sensor') & mut2.startswith('Output'):
                    M = {'As':Mut1_pars[0],'Bs':Mut1_pars[1],'Cs':Mut1_pars[2],'Ns':Mut1_pars[3],'Ar':Mut3_pars[0],'Br':Mut3_pars[1],'Cr':Mut3_pars[2],'Nr':Mut3_pars[3],'Ao':Mut2_pars[0],'Bo':Mut2_pars[1],'Co':Mut2_pars[2],'No':Mut2_pars[3],'Fo':0.0}
                elif mut1.startswith('Sensor') & mut2.startswith('Regulator'):
                    M = {'As':Mut1_pars[0],'Bs':Mut1_pars[1],'Cs':Mut1_pars[2],'Ns':Mut1_pars[3],'Ar':Mut2_pars[0],'Br':Mut2_pars[1],'Cr':Mut2_pars[2],'Nr':Mut2_pars[3],'Ao':Mut3_pars[0],'Bo':Mut3_pars[1],'Co':Mut3_pars[2],'No':Mut3_pars[3],'Fo':0.0}
                elif mut1.startswith('Regulator') & mut2.startswith('Output'):
                    M = {'As':Mut3_pars[0],'Bs':Mut3_pars[1],'Cs':Mut3_pars[2],'Ns':Mut3_pars[3],'Ar':Mut1_pars[0],'Br':Mut1_pars[1],'Cr':Mut1_pars[2],'Nr':Mut1_pars[3],'Ao':Mut2_pars[0],'Bo':Mut2_pars[1],'Co':Mut2_pars[2],'No':Mut2_pars[3],'Fo':0.0}
                elif mut1.startswith('Regulator') & mut2.startswith('Sensor'):
                    M = {'As':Mut2_pars[0],'Bs':Mut2_pars[1],'Cs':Mut2_pars[2],'Ns':Mut2_pars[3],'Ar':Mut1_pars[1],'Br':Mut1_pars[1],'Cr':Mut1_pars[1],'Nr':Mut1_pars[1],'Ao':Mut3_pars[0],'Bo':Mut3_pars[1],'Co':Mut3_pars[2],'No':Mut3_pars[3],'Fo':0.0}
                elif mut1.startswith('Output') & mut2.startswith('Regulator'):
                    M = {'As':Mut3_pars[0],'Bs':Mut3_pars[1],'Cs':Mut3_pars[2],'Ns':Mut3_pars[3],'Ar':Mut2_pars[0],'Br':Mut2_pars[1],'Cr':Mut2_pars[2],'Nr':Mut2_pars[3],'Ao':Mut1_pars[0],'Bo':Mut1_pars[1],'Co':Mut1_pars[2],'No':Mut1_pars[3],'Fo':0.0}
                elif mut1.startswith('Output') & mut2.startswith('Sensor'):
                    M = {'As':Mut2_pars[0],'Bs':Mut2_pars[1],'Cs':Mut2_pars[2],'Ns':Mut2_pars[3],'Ar':Mut3_pars[0],'Br':Mut3_pars[1],'Cr':Mut3_pars[2],'Nr':Mut3_pars[3],'Ao':Mut1_pars[0],'Bo':Mut1_pars[1],'Co':Mut1_pars[2],'No':Mut1_pars[3],'Fo':0.0}
                else:
                    raise KeyError('Mutant names invalid 212')
            
                par_dict = {
                    "A_s":10**WT_pars[0],
                    "B_s":10**WT_pars[1],
                    "C_s":10**WT_pars[2],
                    "N_s":10**WT_pars[3],
                    "MA_s":10**M['As'],
                    "MB_s":10**M['Bs'],
                    "MC_s":10**M['Cs'],
                    "MN_s":10**M['Ns'], 
                    "A_r":10**WT_pars[4],
                    "B_r":10**WT_pars[5],
                    "C_r":10**WT_pars[6],
                    "N_r":WT_pars[7],
                    "MA_r":10**M['Ar'],
                    "MB_r":10**M['Br'],
                    "MC_r":10**M['Cr'],
                    "MN_r":10**M['Nr'],
                    "A_o":10**WT_pars[8],
                    "B_o":10**WT_pars[9],
                    "C_o":10**WT_pars[10],
                    "N_o":10**WT_pars[11],
                    "F_o":10**WT_pars[12],
                    "MA_o":10**M['Ao'],
                    "MB_o":10**M['Bo'],
                    "MC_o":10**M['Co'],
                    "MN_o":10**M['No'],
                    "MF_o":10**M['Fo'],
                        }
            
                par_list = list(par_dict.values())
                parameters.loc[len(parameters)] = par_list
        
        path = f'../results/Triplet_params/{genotypes}.csv'
        count = count + 1
        parameters.to_csv(path, index = False)
        print('mutant ', genotypes, 'completed, ', count, 'out of 1000')

        
# %%
'''Visualise distribution of simulations'''
def Visualise_mut(mutants:list):
    plot_num = 50
    if len(mutants) == 2:
        mut1 = mutants[0]
        mut2 = mutants[1]
        genotype = get_mut_ids(mutants)
        path = f'../results/Combined_mutant_params/Pairwise_params/{genotype}.csv'

        pars_df = pd.read_csv(path)

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
            Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model_muts(I_conc= ind,params_list=par_list)

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
        # axes.set_ylim(,10)
        # axes2.set_ylim(6,10)
        axes.set_ylabel('Log_Fluorescence')
        axes.set_xticks(ticks=range(len(fluo_df.columns)), labels=fluo_df.columns)

    if len(mutants) == 3:
        mut1 = mutants[0]
        mut2 = mutants[1]
        mut3 = mutants[2]
        genotype = get_mut_ids(mutants)
        path = f'../results/Triplet_params/{genotype}.csv'

        pars_df = pd.read_csv(path)

        TM_df = meta_dict['TM']
        triplet_mut_dict = TM_df[TM_df['genotype'] == genotype]

        hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)
     

        set_list = []
        data = meta_dict['WT']
        triplet_inducer = [0.00001, 0.0002, 0.2]
        ind = pd.DataFrame(triplet_inducer)
        low = []
        med = []
        high = []

        for index, row in pars_df.iterrows():
            par_list = row.tolist()
            Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model_muts(I_conc= ind,params_list=par_list)

            low.append(Stripe_est_array.iloc[0,0])
            med.append(Stripe_est_array.iloc[1,0])
            high.append(Stripe_est_array.iloc[2,0])

        data = {'low':np.log10(low), 'medium':np.log10(med), 'high':np.log10(high)}
        fluo_df = pd.DataFrame(data)
        fig, axes = plt.subplots(figsize=(10,6))

        axes2 = axes.twinx()
        point = []
        SD = []
        for obs_m, obs_sd in zip(triplet_mut_dict['obs_fluo_mean'],triplet_mut_dict['obs_SD']):
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
        data_set = mpatches.Patch(color= 'darkcyan', label='Triplet data')
        plt.legend(handles=[data_set,Rand,Wildtype], bbox_to_anchor=(1, 1), title = "Legend")
        plt.title(f'Triplet mutant fit: {genotype}')
        axes.set_xlabel('Inducer Concetration')
        axes.set_ylim(6,10)
        axes2.set_ylim(6,10)
        axes.set_ylabel('Log_Fluorescence')
        axes.set_xticks(ticks=range(len(fluo_df.columns)), labels=fluo_df.columns)

#muts = ['Regulator1', 'Sensor1', 'Output1']
#Visualise_mut(mutants = muts)
#%%
'''Calculating epistasis from params'''

#Edit this to open the correct csv and run the parameters similar to visualise_mut


def pred_fluo(mutants:list, size = 100): 
    '''Takes 2 or 3 mutants in a list and returns a dataframe of 3*size*1000 for predicted fluorescence at low, medium and high inducer concs '''
    if len(mutants) == 2:
        mut1 = mutants[0]
        mut2 = mutants[1]

        genotype = get_mut_ids(mutants)
        path = f'../results/Combined_mutant_params/Pairwise_params/{genotype}.csv'

        pars_df = pd.read_csv(path)

        DM_df = meta_dict['DM']
        pair_mut_dict = DM_df[DM_df['genotype'] == genotype]

        hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)
        # np.random.seed(0)  

        set_list = []
        data = meta_dict['WT']
        pairwise_inducer = [0.00001, 0.0002, 0.2]
        ind = pd.DataFrame(pairwise_inducer)
        low, med, high, low_WT, med_WT, high_WT, low_m1, med_m1, high_m1, low_m2, med_m2, high_m2 = [], [], [], [], [], [], [], [], [], [], [], []

        for index, row in pars_df.iterrows():
            par_list = row.to_list()
            Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model_muts(I_conc= ind,params_list=par_list)

            low.append(Stripe_est_array.iloc[0,0])
            med.append(Stripe_est_array.iloc[1,0])
            high.append(Stripe_est_array.iloc[2,0])

            WT_par_list = row[['A_s','B_s','C_s','N_s','A_r','B_r','C_r','N_r','A_o','B_o','C_o','N_o', 'F_o']].to_list()

            Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model(I_conc= ind,params_list=WT_par_list)

            low_WT.append(Stripe_est_array.iloc[0,0])
            med_WT.append(Stripe_est_array.iloc[1,0])
            high_WT.append(Stripe_est_array.iloc[2,0])

            Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = model_hill.model_single_muts(I_conc= ind,params_list=par_list,mutant=[*mut1][0])

            low_m1.append(Stripe_est_array.iloc[0,0])
            med_m1.append(Stripe_est_array.iloc[1,0])
            high_m1.append(Stripe_est_array.iloc[2,0])

            Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = model_hill.model_single_muts(I_conc= ind,params_list=par_list,mutant=[*mut2][0])

            low_m2.append(Stripe_est_array.iloc[0,0])
            med_m2.append(Stripe_est_array.iloc[1,0])
            high_m2.append(Stripe_est_array.iloc[2,0])

        data = {'low':np.log10(low), 'medium':np.log10(med), 'high':np.log10(high)}
        WT_data = {'low':np.log10(low_WT), 'medium':np.log10(med_WT), 'high':np.log10(high_WT)}
        m1_data = {'low':np.log10(low_m1), 'medium':np.log10(med_m1), 'high':np.log10(high_m1)}
        m2_data = {'low':np.log10(low_m2), 'medium':np.log10(med_m2), 'high':np.log10(high_m2)}
        combo_df = pd.DataFrame(data)
        WT_df = pd.DataFrame(WT_data)
        m1_df = pd.DataFrame(m1_data)
        m2_df = pd.DataFrame(m2_data)
        return combo_df, WT_df, m1_df, m2_df
    elif len(mutants) == 3:
        mut1 = mutants[0]
        mut2 = mutants[1]
        mut3 = mutants[2]


        genotype = get_mut_ids(mutants)
        path = f'../results/Triplet_params/{genotype}.csv'

        pars_df = pd.read_csv(path)

        TM_df = meta_dict['TM']
        triplet_mut_dict = TM_df[TM_df['genotype'] == genotype]

        hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)
     

        set_list = []
        data = meta_dict['WT']
        triplet_inducer = [0.00001, 0.0002, 0.2]
        ind = pd.DataFrame(triplet_inducer)
        low, med, high, low_WT, med_WT, high_WT, low_m1, med_m1, high_m1, low_m2, med_m2, high_m2, low_m3, med_m3, high_m3 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        for index, row in pars_df.iterrows():
            par_list = row.to_list()
            Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model_muts(I_conc= ind,params_list=par_list)

            low.append(Stripe_est_array.iloc[0,0])
            med.append(Stripe_est_array.iloc[1,0])
            high.append(Stripe_est_array.iloc[2,0])

            WT_par_list = row[['A_s','B_s','C_s','N_s','A_r','B_r','C_r','N_r','A_o','B_o','C_o','N_o', 'F_o']].to_list()

            Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model(I_conc= ind,params_list=WT_par_list)

            low_WT.append(Stripe_est_array.iloc[0,0])
            med_WT.append(Stripe_est_array.iloc[1,0])
            high_WT.append(Stripe_est_array.iloc[2,0])

            Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = model_hill.model_single_muts(I_conc= ind,params_list=par_list,mutant=[*mut1][0])

            low_m1.append(Stripe_est_array.iloc[0,0])
            med_m1.append(Stripe_est_array.iloc[1,0])
            high_m1.append(Stripe_est_array.iloc[2,0])

            Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = model_hill.model_single_muts(I_conc= ind,params_list=par_list,mutant=[*mut2][0])

            low_m2.append(Stripe_est_array.iloc[0,0])
            med_m2.append(Stripe_est_array.iloc[1,0])
            high_m2.append(Stripe_est_array.iloc[2,0])

            Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = model_hill.model_single_muts(I_conc= ind,params_list=par_list,mutant=[*mut3][0])

            low_m3.append(Stripe_est_array.iloc[0,0])
            med_m3.append(Stripe_est_array.iloc[1,0])
            high_m3.append(Stripe_est_array.iloc[2,0])

        data = {'low':np.log10(low), 'medium':np.log10(med), 'high':np.log10(high)}
        WT_data = {'low':np.log10(low_WT), 'medium':np.log10(med_WT), 'high':np.log10(high_WT)}
        m1_data = {'low':np.log10(low_m1), 'medium':np.log10(med_m1), 'high':np.log10(high_m1)}
        m2_data = {'low':np.log10(low_m2), 'medium':np.log10(med_m2), 'high':np.log10(high_m2)}
        m3_data = {'low':np.log10(low_m3), 'medium':np.log10(med_m3), 'high':np.log10(high_m3)}
        combo_df = pd.DataFrame(data)
        WT_df = pd.DataFrame(WT_data)
        m1_df = pd.DataFrame(m1_data)
        m2_df = pd.DataFrame(m2_data)
        m3_df = pd.DataFrame(m3_data)
        return combo_df, WT_df, m1_df, m2_df, m3_df


#%%
def Calculate_Eps_hat(mutants:list):
    Eps_hat_low = []
    Eps_hat_medium = []
    Eps_hat_high = []
    if len(mutant) == 2:
        DM, WT, M1, M2 = pred_fluo(mutants) # predicted fluorescence in log10
        #low
        for dm, wt, m1, m2 in zip(DM['low'], WT['low'], M1['low'], M2['low']):
            # Eps_hat = np.log10(10**dm/10**wt) - np.log10((10**m1 * 10**m2)/(10**wt)**2)
            Eps_hat = np.log10(10**dm/10**wt) - np.log10(10**m1/10**wt) + np.log10(10**m2/10**wt)
            Eps_hat_low.append(Eps_hat)
        #medium
        for dm, wt, m1, m2 in zip(DM['medium'], WT['medium'], M1['medium'], M2['medium']):
            # Eps_hat = np.log10(10**dm/10**wt) - np.log10((10**m1 * 10**m2)/(10**wt)**2)
            Eps_hat = np.log10(10**dm/10**wt) - np.log10(10**m1/10**wt) + np.log10(10**m2/10**wt)
            Eps_hat_medium.append(Eps_hat)
        #high
        for dm, wt, m1, m2 in zip(DM['high'], WT['high'], M1['high'], M2['high']):
            # Eps_hat = np.log10(10**dm/10**wt) - np.log10((10**m1 * 10**m2)/(10**wt)**2)
            Eps_hat = np.log10(10**dm/10**wt) - np.log10(10**m1/10**wt) + np.log10(10**m2/10**wt)
            Eps_hat_high.append(Eps_hat)

    if len(mutant) == 3:
        TM, WT, M1, M2, M3 = pred_fluo(mutants) # predicted fluorescence in log10
        #low
        for tm, wt, m1, m2, m3 in zip(TM['low'], WT['low'], M1['low'], M2['low'], M3['low']):
            Eps_hat = np.log10(10**tm/10**wt) - np.log10((10**m1 * 10**m2 * 10**m3)/(10**wt)**3)
            #Eps_hat = np.log10(10**dm/10**wt) - np.log10(10**m1/10**wt) + np.log10(10**m2/10**wt)
            Eps_hat_low.append(Eps_hat)
        #medium
        for tm, wt, m1, m2, m3 in zip(TM['medium'], WT['medium'], M1['medium'], M2['medium'], M3['medium']):
            Eps_hat = np.log10(10**tm/10**wt) - np.log10((10**m1 * 10**m2 * 10**m3)/(10**wt)**3)
            #Eps_hat = np.log10(10**dm/10**wt) - np.log10(10**m1/10**wt) + np.log10(10**m2/10**wt)
            Eps_hat_medium.append(Eps_hat)
        #high
        for tm, wt, m1, m2, m3 in zip(TM['high'], WT['high'], M1['high'], M2['high'], M3['high']):
            Eps_hat = np.log10(10**tm/10**wt) - np.log10((10**m1 * 10**m2 * 10**m3)/(10**wt)**3)
            #Eps_hat = np.log10(10**dm/10**wt) - np.log10(10**m1/10**wt) + np.log10(10**m2/10**wt)
            Eps_hat_high.append(Eps_hat)

    df = pd.DataFrame({'Low_eh':Eps_hat_low, 'medium_eh': Eps_hat_medium, 'high_ep': Eps_hat_high})
    return df

def Visualise_EPs_mode(prior_mutant):
    DM_names = DM_stripes['genotype'].tolist()
    DM_names = list(set(DM_names[3:]))
    DM_names.sort()
    count = 0
    Eps_list = []
    if prior_mutant == None:
        Eps_mode_df = pd.DataFrame({'Genotype': [], 'low':[],'medium': [], 'high':[]})
        path = '../results/Eps_hat_pairwise/old/Pairwise_Eps_hat_mode.csv'
        mutant_range:slice=slice(0,len(DM_names))
        Eps_mode_df.to_csv(path, index = False)
    else:
        position = DM_names.index(prior_mutant)
        mutant_range:slice=slice(position+1,len(DM_names))
        path = '../results/Eps_hat_pairwise/old/Pairwise_Eps_hat_mode.csv'
        
    for i, genotypes in enumerate(DM_names[mutant_range]):
        Low_list, Medium_list, High_list, Genotype_list = [], [], [], []

        Genotype_df = pd.read_csv(f"../results/Eps_hat_pairwise/old/{genotypes}.csv")

        # Genotype_df = df[df.Genotype.isin([f"{genotypes}"])]
        
        l,m,h = Kde_mode(Genotype_df) 

        Eps_list.append(l)
        Eps_list.append(m)
        Eps_list.append(h)
        #append low med high modes to a dataframe with corresponding genotype name. 

        df = pd.DataFrame({'Genotype': [genotypes], 'low':[l],'medium': [m], 'high':[h]})
        df.to_csv(path,mode='a', header=False, index=False)
        count += 1
        print(f'mutant {genotypes}', count)
    print(f'mutants complete')
    return Eps_list

#Eps_list = Visualise_EPs_mode(prior_mutant=None)
plt.hist(Eps_list, bins = 'auto', density = True)
plt.xlabel('Epistasis_hat')
plt.ylabel('density')
plt.title(f'Epistasis_hat old (data) method')

df = pd.read_csv('../results/Eps_hat_pairwise/Pairwise_Eps_hat_mode.csv')
Eps_list = []
for l, m, h in zip(df['low'], df['medium'], df['high']):
    Eps_list.append(l)
    Eps_list.append(m)
    Eps_list.append(h)
plt.hist(Eps_list, bins = 'auto', density = True)
plt.xlabel('Epistasis_hat')
plt.ylabel('density')
plt.title(f'Epistasis_hat new (model-only) method')





#%%
#old method

# def old_predicted_fluo(mutants):
#     mut1 = mutants[0]
#     mut2 = mutants[1]

#     genotype = get_mut_ids(mutants)
#     path = f'../results/Combined_mutant_params/Pairwise_params/{genotype}.csv'

#     pars_df = pd.read_csv(path)

#     DM_df = meta_dict['DM']
#     pair_mut_dict = DM_df[DM_df['genotype'] == genotype]

#     hill=model_hill(params_list=[1]*13,I_conc=meta_dict["WT"].S)
#     # np.random.seed(0)  

#     set_list = []
#     data = meta_dict['WT']
#     pairwise_inducer = [0.00001, 0.0002, 0.2]
#     ind = pd.DataFrame(pairwise_inducer)
#     low, med, high = [], [], []

#     for index, row in pars_df.iterrows():
#         par_list = row.to_list()
#         Sensor_est_array,Regulator_est_array,Output_est_array, Stripe_est_array = hill.model_muts(I_conc= ind,params_list=par_list)

#         low.append(Stripe_est_array.iloc[0,0])
#         med.append(Stripe_est_array.iloc[1,0])
#         high.append(Stripe_est_array.iloc[2,0])
#     return low, med, high

# def old_Eps_hat(prior_mutant):
#     DM_names = DM_stripes['genotype'].tolist()
#     DM_names = list(set(DM_names[3:]))
#     DM_names.sort()
#     count = 0
#     Eps_hat_low = []
#     Eps_hat_medium = []
#     Eps_hat_high = []
    
#     if prior_mutant == None:
#         mutant_range:slice=slice(0,len(DM_names))
#     else:
#         position = DM_names.index(prior_mutant)
#         mutant_range:slice=slice(position+1,len(DM_names))
        
#     df_DM = meta_dict['DM']
#     g_WT = np.array(df_DM['obs_fluo_mean'][df_DM['genotype']=='WT']) #wildtype fluorescence

#     for i, genotypes in enumerate(DM_names[mutant_range]):
#         muts = get_mut_names(genotypes) 
#         low, medium, high = old_predicted_fluo(muts) #obtain predicted fluorescence for duplex only
#         Glog= G_log(muts)[0] #obtain logged relative fluorescence from data.
#         for l, m, h in zip(low,medium,high):
#             Eps_low_hat =  np.log10(l/g_WT[0]) - Glog[0] #calculated epistasis hat
#             Eps_hat_low.append(Eps_low_hat)
#             Eps_med_hat =  np.log10(m/g_WT[1]) - Glog[1] 
#             Eps_hat_medium.append(Eps_med_hat)
#             Eps_high_hat =  np.log10(h/g_WT[2]) - Glog[2] 
#             Eps_hat_high.append(Eps_high_hat)


#         df = pd.DataFrame({'Low_eh':Eps_hat_low, 'medium_eh': Eps_hat_medium, 'high_ep': Eps_hat_high})
#         path = f'../results/Eps_hat_pairwise/old/{genotypes}.csv'
#         df.to_csv(path, index = False)
#         print(f'{genotypes} complete, {i+1} out of 300')
#     return

def Calculate_Eps_old(mutants):
    DM, WT, M1, M2 = pred_fluo(mutants) #fluorescence in log10
    df_DM = meta_dict['DM'] 
    Glog= G_log(mutants)[0] #WT fluorescence at each inducer
    g_WT = np.array(df_DM['obs_fluo_mean'][df_DM['genotype']=='WT']) #wildtype fluorescence
    Eps_low_hat, Eps_med_hat, Eps_high_hat = [], [], []
    for l,m,h in zip(DM['low'],DM['medium'],DM['high']):
         Eps_low_hat.append(np.log10(10**l/g_WT[0]) - Glog[0])
         Eps_med_hat.append(np.log10(10**m/g_WT[1]) - Glog[1])
         Eps_high_hat.append(np.log10(10**h/g_WT[2]) - Glog[2])
    df = pd.DataFrame({'low_eh':Eps_low_hat, 'medium_eh': Eps_med_hat, 'high_eh': Eps_high_hat})
    return df
    
def Calculate_Eps_old_all(prior_mutant):
    DM_names = DM_stripes['genotype'].tolist()
    DM_names = list(set(DM_names[3:]))
    DM_names.sort()

    if prior_mutant == None:
        mutant_range:slice=slice(0,len(DM_names))
    else:
        position = DM_names.index(prior_mutant)
        mutant_range:slice=slice(position+1,len(DM_names))

    for i, genotypes in enumerate(DM_names[mutant_range]):
        muts = get_mut_names(genotypes) 
        df = Calculate_Eps_old(muts)
        df = pd.DataFrame({'genotype':[genotypes]*len(df['low_eh']),'low_eh':df['low_eh'], 'medium_eh': df['medium_eh'], 'high_eh': df['high_eh']})
        path = f'../results/Eps_hat_pairwise/old/{genotypes}.csv'
        df.to_csv(path, index = False)
        print(f'{genotypes} complete, {i+1} out of ', len(DM_names[mutant_range]))
    return

    











































#%%

def Eps(df, Glog):
    Eps_low = []
    Eps_med = []
    Eps_high = []
    for l,m,h in zip(df['G_low'],df['G_medium'],df['G_high']):
        Ep_low = l - Glog[0]
        Eps_low.append(Ep_low)

        Ep_med = m - Glog[1]
        Eps_med.append(Ep_med)

        Ep_high = h - Glog[2]
        Eps_high.append(Ep_high)
    
    genotype = df['Genotype'].tolist()

    temp = {'Genotype':genotype, 'Eps_low':Eps_low, 'Eps_medium':Eps_med, 'Eps_high':Eps_high}

    df2 = pd.DataFrame(temp)

    return df2

def Generate_Eps_hat(mutant_list):
        '''Get epistasis values for a given fluroescence output per mutant '''
        #find G_log of mutants of interest
        Glog_mean= G_log(mutant_list)[0]

        #Get G_hat values for all predicted fluorescence 
        LMH_df = obtain_pred_fluo(mutant_list) #low,med,high fluoresence
        df = G_hat_all(LMH_df)
        
        return Eps(df,Glog_mean), Glog_mean

def Generate_Epshat_all_DM(prior_mutant:None):


    '''Create large dataframe of all Epistasis at low, medium and high inducer concs'''
    DM_names = DM_stripes['genotype'].tolist()
    DM_names = list(set(DM_names[3:]))
    
    if prior_mutant == None:
        Eps_pairwise_df = pd.DataFrame({'Genotype': [], 'Category':[], 'Epistasis_hat':[], 'Epistasis_obs': []})
        mutant_range:slice=slice(0,len(DM_names))
        Eps_pairwise_df.to_csv('../results/Pairwise_Eps.csv', index = False)
        path = f"../results/Pairwise_Eps.csv"
    else:
        position = DM_names.index(prior_mutant)
        path = f"../results/Pairwise_Eps.csv"
        Eps_pairwise_df = pd.read_csv(path)
        mutant_range:slice=slice(position,len(DM_names))

    for genotypes in DM_names[mutant_range]:
        #get genotypeID
        mutant_list = get_mut_names(genotypes)
        #Obtain epistasis values for predictions and LAD from LMH data 
        Eps_df, G_log_mean= Generate_Eps_hat(mutant_list)
        
        G = G_lab(mutant_list)
        G_obs = G[0] #observed fluoresence
        low_Epsilon =  G_obs[0] - G_log_mean[0] #G_log_mean from the LAD model
        med_Epsilon =  G_obs[1] - G_log_mean[1]
        high_Epsilon =  G_obs[2] - G_log_mean[2]

        Eps_hat = [] 
        Categories = []
        Eps_obs = []

        for items in Eps_df['Eps_low'].tolist():
            Eps_hat.append(items)
            Categories.append('low')
            Eps_obs.append(low_Epsilon)
        for items in Eps_df['Eps_medium'].tolist():
            Eps_hat.append(items)
            Categories.append('medium')
            Eps_obs.append(med_Epsilon)
        for items in Eps_df['Eps_high'].tolist():
            Eps_hat.append(items)
            Categories.append('high')
            Eps_obs.append(high_Epsilon)

        Genotype_list = Eps_df['Genotype'].tolist() * 3


        temp = {'Genotype': Genotype_list, 'Category':Categories, 'Epistasis_hat':Eps_hat, 'Epistasis_obs': Eps_obs}

        new_df = pd.DataFrame(temp)
        new_df.to_csv(path,mode='a', header=False, index=False)

        print(f'Mutant {genotypes} completed')
    
    print('mutants complete')
    return

def Generate_Epshat_all_TM(prior_mutant:None):
    '''Create large dataframe of all Epistasis at low, medium and high inducer concs'''
    TM_names = TM_stripes['genotype'].tolist()
    TM_names = list(set(TM_names[3:]))
    TM_names.sort()
    
    if prior_mutant == None:
        Eps_triplet_df = pd.DataFrame({'Genotype': [], 'Category':[], 'Epistasis_hat':[], 'Epistasis_obs': []})
        mutant_range:slice=slice(0,len(TM_names))
        Eps_triplet_df.to_csv('../results/Triplet_Eps.csv', index = False)
        path = f"../results/Triplet_Eps.csv"
    else:
        position = TM_names.index(prior_mutant)
        path = f"../results/Triplet_Eps.csv"
        Eps_triplet_df = pd.read_csv(path)
        mutant_range:slice=slice(position+1,len(TM_names))

    for genotypes in TM_names[mutant_range]:
        #get genotypeID
        mutant_list = get_mut_names(genotypes)
        #Obtain epistasis values for predictions and LAD from LMH data 
        Eps_df, G_log_mean= Generate_Eps_hat(mutant_list)
        
        G = G_lab(mutant_list)
        G_obs = G[0] #observed fluoresence
        low_Epsilon =  G_obs[0] - G_log_mean[0] #G_log_mean from the LAD model
        med_Epsilon =  G_obs[1] - G_log_mean[1]
        high_Epsilon =  G_obs[2] - G_log_mean[2]

        Eps_hat = [] 
        Categories = []
        Eps_obs = []

        for items in Eps_df['Eps_low'].tolist():
            Eps_hat.append(items)
            Categories.append('low')
            Eps_obs.append(low_Epsilon)
        for items in Eps_df['Eps_medium'].tolist():
            Eps_hat.append(items)
            Categories.append('medium')
            Eps_obs.append(med_Epsilon)
        for items in Eps_df['Eps_high'].tolist():
            Eps_hat.append(items)
            Categories.append('high')
            Eps_obs.append(high_Epsilon)

        Genotype_list = Eps_df['Genotype'].tolist() * 3


        temp = {'Genotype': Genotype_list, 'Category':Categories, 'Epistasis_hat':Eps_hat, 'Epistasis_obs': Eps_obs}

        new_df = pd.DataFrame(temp)
        new_df.to_csv(path,mode='a', header=False, index=False)

        # book = load_workbook(path)
        # writer = pd.ExcelWriter(path, engine='openpyxl')
        # writer.book = book
        # Eps_pairwise_df = pd.read_excel(path)
        # updated_data = pd.concat([Eps_pairwise_df,new_df],ignore_index=True)
        # updated_data.to_excel(writer, index = False)
        print(f'Mutant {genotypes} completed')
    
    print('mutants complete')
    return


#%%
'''Visualising distribution'''

