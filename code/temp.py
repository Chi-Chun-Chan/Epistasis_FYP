from Epistasis import *

DM_names = DM_stripes['genotype'].tolist()
DM_names = list(set(DM_names[3:]))
DM_names.sort()
count = 0
Eps_list = []

mutant_range:slice=slice(0,len(DM_names))

All_Eps_old = pd.DataFrame()
for i, genotypes in enumerate(DM_names[mutant_range]):
        Low_list, Medium_list, High_list, Genotype_list = [], [], [], []

        Genotype_df = pd.read_csv(f"../results/Eps_hat_pairwise/old/{genotypes}.csv")
        All_Eps_old = All_Eps_old.append(Genotype_df['low_eh'].to_list())
        All_Eps_old = All_Eps_old.append(Genotype_df['medium_eh'].to_list())
        All_Eps_old = All_Eps_old.append(Genotype_df['high_eh'].to_list())
plt.hist(All_Eps_old, bins='auto', density=True)
plt.xlabel('Epistasis_hat')
plt.ylabel('density')
plt.title(f'Epistasis_hat old (data) method')

All_Eps = pd.DataFrame()
for i, genotypes in enumerate(DM_names[mutant_range]):
        Low_list, Medium_list, High_list, Genotype_list = [], [], [], []

        Genotype_df = pd.read_csv(f"../results/Eps_hat_pairwise/{genotypes}.csv")

        
        All_Eps = All_Eps.append(Genotype_df['low_hat'].to_list())
        All_Eps = All_Eps.append(Genotype_df['medium_hat'].to_list())
        All_Eps = All_Eps.append(Genotype_df['high_hat'].to_list())
plt.hist(All_Eps, bins='auto', density=True)
plt.xlabel('Epistasis_hat')
plt.ylabel('density')
plt.title(f'Epistasis_hat new (model-only) method')

Eps_df = pd.read_csv(f"../results/Eps_hat_pairwise/old/Pairwise_Eps_hat_mode.csv")
cols = Eps_df.columns.values
Eps_list = [*Eps_df[cols[1]],*Eps_df[cols[2], *Eps_df[cols[3]]]]

#visualise distribution of eps_hat
mut = 'R7_O7'
test_df = pd.read_csv(f"../results/Eps_hat_pairwise/old/{mut}.csv")
plt.hist(test_df['low_eh'], bins = 'auto', density = True)
plt.xlabel('Epistasis_hat')
plt.ylabel('density')
plt.title(f'{mut} Epistasis')

