from data_wrangling import meta_dict
from Models import *
from Epistasis_calc_functions import *

#This file is used to calculate epistasis for all double and triple mutants from observed and expected flourcsence at low, medium and high inducer concs

#first, get the parameters calculated in model_fit.py
#df_fits = get_params()
################################################################
#run here to get excel spreadsheet of epistases for a given model a datframe called df_Eps, default is 'observed' which calculates epistasis w.r.t. lab data
#file path for model called 'MODEL' and sampling strateygy 'STRATEGY' will be '../results/Eps_MODEL_STRATEGY.xlsx' 
#Observed epistasis is stored as '../results/Eps_observed.xlsx' 
# & get a dataframe comparing inducer dependent epistasis from df_Ep
df_Eps1 = Eps_toExcel(model = model_hill, strategy = 'all')
#################################################
df = pd.read_excel('../data/Source_Data.xlsx', sheet_name='Figure 2')

Eps_Rub = []
Eps_Us = []
for mut in df_Eps1['genotype']:
    Eps_Rub.append(list(df['Unnamed: 9'][df['Unnamed: 0'].isin([mut])]))
    Eps_Us.append(list(df_Eps1['Ep'][df_Eps1['genotype'].isin([mut])]))

x = Eps_Rub
y = Eps_Us
import matplotlib.pyplot as plt
plt.scatter(x,y)

# AK
# Mann-Whitney U test does not assume normality
# changes were made to Epistasis function 

# check output type for p value for T-test for means of two independent samples
mutants = ['Output1', 'Regulator1']
G = G_obs(mutants)
G_obs_mean = G[0]
G_obs_std = G[1]
Ghat_logadd = G_log(mutants)
Ghat_logadd_mean = Ghat_logadd[0]
Ghat_logadd_std = Ghat_logadd[1]
Epsilon =  G[0] - Ghat_logadd[0]
ttest_p_val = stats.ttest_ind_from_stats(mean1 = Ghat_logadd_mean,
std1 = Ghat_logadd_std, nobs1 = 3,mean2 = G_obs_mean, std2 = G_obs_std,
nobs2 = 3)[1]
ttest_p_val
# check output p value for Mann-Whitney U test
mann_p_val_object = stats.mannwhitneyu(Ghat_logadd_mean, G_obs_mean, alternative='two-sided')
mann_p_val = getattr(mann_p_val_object, 'pvalue') 
mann_p_val

# histograms

# histogram for p values
# better distribution of p values
df_Eps["pVal_epistasis"].hist()
plt.title("Histogram of p values")

# check if generated p values are equal to the p values from initial data
copy = df_Eps
copy1 = copy.dropna(subset=['pVal_epistasis'])
output_pval =  list(copy1["pVal_epistasis"])
n_pval = len(output_pval)
check_data = pd.ExcelFile('../data/Source_Data.xlsx')
check_df = pd.read_excel(check_data, 'Figure 2', header = 1, usecols="K")
checkdf = check_df.dropna(subset=['p_value'])
check_pval = list(checkdf['p_value'])
n_check = len(check_pval)
if output_pval == check_pval:
    print("True")
else:
    print("False")